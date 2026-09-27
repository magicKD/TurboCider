#include "models/ltx_runtime/ltx_streaming_plan.hpp"
#include "models/ltx_runtime/ltx_gemma_encoder.h"
#include "models/ltx_runtime/ltx_weights.h"
#include "runtime/session.hpp"
#include "runtime/streaming/resolved_request.hpp"

#include <atomic>
#include <cassert>
#include <cstring>
#include <filesystem>
#include <fcntl.h>
#include <fstream>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>
#include <unistd.h>

namespace tc {
std::unique_ptr<ModelSession> create_ltx_native_candidate(
    const std::filesystem::path &);
}

namespace {

tc::StreamingConfig config() {
    tc::StreamingConfig value;
    value.enabled = true;
    value.schema_version = 1;
    value.selection = "manual";
    value.retention = "request";
    value.stages["ltx-stage1-denoiser"] = {
        "streamed", 1, 2, 8, 1, 2};
    value.stages["ltx-stage2-denoiser"] = {
        "streamed", 1, 2, 8, 1, 2};
    return value;
}

tc::Request request() {
    tc::Request value;
    value.model = "ltx-2.5-distilled";
    value.operation = "video.generate";
    value.prompt = "g";
    value.output = "/tmp/not-generated-ltx-public.mp4";
    value.execution = "gpu";
    value.ltx_backend = "c_metal";
    value.width = 64;
    value.height = 64;
    value.frames = 9;
    value.fps = 24;
    value.steps = 11;
    value.audio = false;
    value.dynamic_text = true;
    value.ltx_fast_av = true;
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

void verify_gemma_fd_authority(const std::filesystem::path &root) {
    const auto named = root / "gemma-fd-authority.safetensors";
    {
        std::ofstream stream(named, std::ios::binary);
        const uint64_t header_bytes = 2;
        stream.write(reinterpret_cast<const char *>(&header_bytes),
                     sizeof(header_bytes));
        stream.write("{}", 2);
    }
    const int descriptor = ::open(named.c_str(), O_RDONLY | O_CLOEXEC);
    assert(descriptor >= 0);
    const auto held = named.string() + ".held";
    std::filesystem::rename(named, held);
    {
        std::ofstream replacement(named, std::ios::binary);
        replacement << "replacement";
    }

    char error[1024] = {};
    ltx_gemma_checkpoint_info info{};
    assert(!ltx_gemma_checkpoint_inspect_fd(
        descriptor, named.c_str(), &info, error, sizeof(error)));
    assert(std::strstr(error, "no gemma_config metadata") != nullptr);
    std::memset(error, 0, sizeof(error));
    assert(!ltx_gemma_checkpoint_inspect(
        named.c_str(), &info, error, sizeof(error)));
    assert(std::strstr(error, "no gemma_config metadata") == nullptr);

    ltx_gemma_encoder_options options{};
    options.checkpoint = named.c_str();
    options.tokenizer_json = named.c_str();
    options.shader_source = named.c_str();
    options.source_fd_version = 1;
    options.checkpoint_fd = descriptor;
    options.tokenizer_fd = descriptor;
    std::memset(error, 0, sizeof(error));
    assert(ltx_gemma_encoder_create(&options, error, sizeof(error)) == nullptr);
    assert(std::strstr(error, "no gemma_config metadata") != nullptr);
    ::close(descriptor);
}

} // namespace

int main(int argc, char **argv) {
    assert(argc == 2);
    try {
        verify_gemma_fd_authority(argv[1]);
        auto session = tc::create_ltx_native_candidate(argv[1]);
        const auto base_request = request();
        const auto probe = session->probe_public_streaming(
            {base_request, device(), "ltx_worker"});
        assert(probe && probe->model_id() == "ltx-2.5-distilled");
        assert(probe->source_lease() != nullptr);
        assert(probe->source_lease()->file_count() == 8);
        assert(probe->workload_identity().width == 64);
        assert(probe->workload_identity().height == 64);
        assert(probe->workload_identity().frames == 9);
        assert(probe->workload_identity().fps == 24);
        assert(probe->workload_identity().steps == 11);
        assert(probe->workload_identity().token_shapes.size() == 1);
        assert(probe->workload_identity().token_shapes.front().compute_rows ==
               1024);

        const auto value_probe = std::dynamic_pointer_cast<
            const tc::streaming::ValueModelStreamingProbe>(probe);
        assert(value_probe && value_probe->lease_ptr().get() ==
                                  probe->source_lease());
        const tc::ltx::StreamingWorkload workload{
            64, 64, 9, 24, 1024, true, true, false, "connected", true};
        const tc::ltx::StreamingPlanView expected(
            value_probe->lease_ptr(),
            "diffusion_models/ltx-2.5-22b-distilled-transformer-comfy-int8-convrot.safetensors",
            config(), workload, value_probe->lease().generation());
        assert(expected.layout().stages.size() == 2);
        assert(expected.layout().stages[0].id == "ltx-stage1-denoiser");
        assert(expected.layout().stages[0].pass_count == 8);
        assert(expected.layout().stages[1].id == "ltx-stage2-denoiser");
        assert(expected.layout().stages[1].pass_count == 3);
        assert(expected.native_stage_options().schedule_pass_begin == 0);
        const tc::ltx::StreamingPlanView expected_stage2(
            value_probe->lease_ptr(),
            "diffusion_models/ltx-2.5-22b-distilled-transformer-comfy-int8-convrot.safetensors",
            config(), workload, value_probe->lease().generation(), 1);
        assert(expected_stage2.layout().digest == expected.layout().digest);
        assert(expected_stage2.native_stage_options().schedule_pass_begin == 8);

        tc::streaming::StreamingPresetRecord record;
        record.id = "ltx-public-host-test";
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
                {unsupported, device(), "ltx_worker"});
        }, "streaming_route_unsupported");
        unsupported = base_request;
        unsupported.ane_manifest = "/tmp/not-opened-ltx-ane.json";
        rejects([&] {
            session->probe_public_streaming(
                {unsupported, device(), "ltx_worker"});
        }, "streaming_route_unsupported");

        tc::streaming::ResolvedStreamingSelection selection;
        selection.record = record;
        selection.device = device();
        selection.exact_selector.enabled = true;
        selection.exact_selector.selection = "preset";
        selection.exact_selector.target_request_memory_bytes = 8ull << 30;
        auto resolved_request = base_request;
        resolved_request.streaming = config();
        resolved_request.output = "/tmp/not-generated-ltx-public.png";
        auto execution = std::make_shared<
            const tc::streaming::ResolvedRequestExecution>(
                tc::streaming::ResolvedRequestExecution{
                    resolved_request, selection, probe, snapshot,
                    "host-test-request"});
        std::atomic<bool> cancelled{false};
        const tc::Event event = [](const std::string &, int, int) {};
        rejects([&] {
            session->generate_resolved(execution, event, cancelled);
        }, "LTX output must be .mp4");
        rejects([&] {
            session->generate_resolved(execution, event, cancelled);
        }, "LTX output must be .mp4");

        const auto transformer = std::filesystem::path(argv[1]) /
            "diffusion_models/ltx-2.5-22b-distilled-transformer-comfy-int8-convrot.safetensors";
        const auto moved = transformer.string() + ".moved";
        std::filesystem::rename(transformer, moved);
        {
            std::ofstream replacement(transformer, std::ios::binary);
            replacement << "replacement";
        }
        rejects([&] { snapshot->revalidate_source(); }, "source");

        std::cout << "PASS LTX public adapter: eight-artifact SourceLease, "
                     "fd Gemma/tokenizer/descriptor, exact identity/layout, route "
                     "rejection, failure cleanup and source replacement "
                     "detection\n";
    } catch (const std::exception &error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
    return 0;
}
