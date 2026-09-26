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
    assert(argc == 4);
    try {
        tc::ZImage session(argv[1]);
        const auto base_request = request();
        tc::streaming::PublicResolveInput input{
            base_request, device(), "embedded_app"};
        const auto probe = session.probe_public_streaming(input);
        assert(probe && probe->model_id() == "z-image-turbo");
        assert(probe->source_lease() != nullptr);
        assert(probe->source_lease()->file_count() == 4);
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
            256, 256, (token.valid_rows + 31) / 32 * 32, 3};
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

        // Qwen encoder padding is removed before the DiT caption is aligned
        // to 32 rows. Fixed encoder length must not inflate the denoiser plan.
        assert(token.padded_rows == token.valid_rows);
        assert(token.compute_rows == token.valid_rows);
        auto fixed_request = base_request;
        fixed_request.dynamic_text = false;
        const auto fixed_probe = session.probe_public_streaming(
            {fixed_request, device(), "embedded_app"});
        const auto &fixed_token = fixed_probe->workload_identity().token_shapes.front();
        assert(fixed_token.valid_rows == token.valid_rows);
        assert(fixed_token.padded_rows == 512 && fixed_token.compute_rows == 512);
        assert(fixed_probe->workload_identity() != probe->workload_identity());
        auto fixed_record = record;
        fixed_record.workload = fixed_probe->workload_identity();
        const auto fixed_snapshot = session.compile_public_streaming(fixed_probe, fixed_record);
        assert(fixed_snapshot->layout().digest == snapshot->layout().digest);

        auto long_request = fixed_request;
        long_request.prompt = std::string(600, 'g');
        const auto long_probe = session.probe_public_streaming(
            {long_request, device(), "embedded_app"});
        const auto &long_token = long_probe->workload_identity().token_shapes.front();
        assert(long_token.valid_rows > 512 && long_token.valid_rows <= 1024);
        assert(long_token.padded_rows == long_token.valid_rows);
        assert(long_token.compute_rows == long_token.valid_rows);
        const tc::z_image::StreamingPlanView long_expected(
            value_probe->lease_ptr(), config(),
            {256, 256, (long_token.valid_rows + 31) / 32 * 32, 3});
        auto long_record = record;
        long_record.workload = long_probe->workload_identity();
        long_record.plan.layout_digest = long_expected.layout().digest;
        const auto long_snapshot = session.compile_public_streaming(long_probe, long_record);
        assert(long_snapshot->layout().digest == long_expected.layout().digest);

        tc::ZImage shared(argv[3]);
        const auto shared_probe = shared.probe_public_streaming(input);
        const auto *shared_lease = shared_probe->source_lease();
        assert(shared_lease && shared_lease->file_count() == 6);
        assert(shared_lease->file("text_encoder/model-00001-of-00002.safetensors").bytes == 14);
        assert(shared_lease->file("text_encoder/model-00002-of-00002.safetensors").bytes == 14);
        rejects([&] { shared_lease->file("text_encoder/model.safetensors"); }, "logical id");
        auto shared_record = record;
        shared_record.source = shared_probe->source_identity();
        auto shared_value = std::dynamic_pointer_cast<const tc::streaming::ValueModelStreamingProbe>(shared_probe);
        tc::z_image::StreamingPlanView shared_plan(shared_value->lease_ptr(), config(), workload);
        shared_record.plan.layout_digest = shared_plan.layout().digest;
        const auto shared_snapshot = shared.compile_public_streaming(shared_probe, shared_record);
        shared_snapshot->revalidate_source();
        const auto shared_binding = std::filesystem::path(argv[3]) / "text_encoder";
        const auto replacement_dir = std::filesystem::path(argv[3]) / "replacement-text";
        std::filesystem::copy(std::filesystem::canonical(shared_binding), replacement_dir,
                              std::filesystem::copy_options::recursive | std::filesystem::copy_options::copy_symlinks);
        std::filesystem::remove(shared_binding);
        std::filesystem::create_directory_symlink(replacement_dir, shared_binding);
        rejects([&] { shared_snapshot->revalidate_source(); }, "source path");

        // A checkpoint filename is not a weight-format capability. A renamed
        // ConvRot source must never enter the BF16 public receipt/catalog path.
        tc::ZImage renamed(argv[2]);
        const auto quantized_probe = renamed.probe_public_streaming(input);
        auto quantized_record = record;
        quantized_record.source = quantized_probe->source_identity();
        quantized_record.workload = quantized_probe->workload_identity();
        quantized_record.runtime = quantized_probe->runtime_identity();
        rejects([&] { renamed.compile_public_streaming(quantized_probe, quantized_record); },
                "requires BF16 tensor metadata");

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

        // A live session's constructor tokenizer must not determine a new
        // public probe after the installed tokenizer generation changes.
        const auto tokenizer_path = std::filesystem::path(argv[1]) / "tokenizer/tokenizer.json";
        auto tokenizer_fd = value_probe->lease_ptr()->duplicate_fd("tokenizer");
        const auto tokenizer_bytes = value_probe->lease_ptr()->file("tokenizer").bytes;
        tc::Tokenizer original_tokenizer(tokenizer_fd.get(), tokenizer_bytes);
        assert(original_tokenizer.raw("g").ids == std::vector<int>{103});
        const auto moved_tokenizer = tokenizer_path.string() + ".moved";
        std::filesystem::rename(tokenizer_path, moved_tokenizer);
        std::ifstream original_file(moved_tokenizer);
        std::string tokenizer_json((std::istreambuf_iterator<char>(original_file)), {});
        const auto token_offset = tokenizer_json.find("\"g\": 103");
        assert(token_offset != std::string::npos);
        tokenizer_json.replace(token_offset, 8, "\"g\": 104");
        { std::ofstream replacement(tokenizer_path); replacement << tokenizer_json; }
        tc::Tokenizer held_tokenizer(tokenizer_fd.get(), tokenizer_bytes);
        assert(held_tokenizer.raw("g").ids == std::vector<int>{103});
        tc::Tokenizer replacement_tokenizer(tokenizer_path.parent_path());
        assert(replacement_tokenizer.raw("g").ids == std::vector<int>{104});
        rejects([&] { snapshot->revalidate_source(); }, "source path");
        const auto replacement_probe = session.probe_public_streaming(input);
        assert(replacement_probe->source_identity() != probe->source_identity());
        const auto replacement_snapshot = [&] {
            auto replacement_record = record;
            replacement_record.source = replacement_probe->source_identity();
            auto replacement_value = std::dynamic_pointer_cast<
                const tc::streaming::ValueModelStreamingProbe>(replacement_probe);
            tc::z_image::StreamingPlanView plan(replacement_value->lease_ptr(), config(), workload);
            replacement_record.plan.layout_digest = plan.layout().digest;
            return session.compile_public_streaming(replacement_probe, replacement_record);
        }();
        rejects([&] { tc::Tokenizer invalid(-1, tokenizer_bytes); }, "invalid tokenizer source size");
        rejects([&] { tc::Tokenizer truncated(tokenizer_fd.get(), tokenizer_bytes + 1); }, "truncated");

        const auto transformer = std::filesystem::path(argv[1]) /
            "split_files/diffusion_models/z_image_turbo_bf16.safetensors";
        const auto moved = transformer.string() + ".moved";
        std::filesystem::rename(transformer, moved);
        {
            std::ofstream replacement(transformer, std::ios::binary);
            replacement << "replacement";
        }
        rejects([&] { replacement_snapshot->revalidate_source(); },
                "source path");

        std::cout << "PASS Z-Image public adapter: single-file and shared sharded text leases, "
                     "exact identity/layout snapshot, route rejection, target "
                     "failure cleanup and source replacement detection\n";
    } catch (const std::exception &error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
    return 0;
}
