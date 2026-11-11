/* Reuse the independent integer oracles and tensor helpers. No full model. */
#define main ltx_existing_w8a8_unit_main
#include "ltx_w8a8_test.mm"
#undef main

int main(int argc, char **argv) {
    @autoreleasepool { try {
        check(argc == 2);
        auto gpu = std::unique_ptr<ltx_gpu, decltype(&ltx_gpu_free)>(
            ltx_gpu_create(argv[1], error, sizeof(error)), ltx_gpu_free);
        check(gpu != nullptr);
        if (!ltx_gpu_convrot_w8a8_info(gpu.get(), nullptr, nullptr)) {
            std::puts("SKIP stage workspace: W8A8 unavailable");
            return 0;
        }
        for (uint32_t bad_rows : {0u, 4097u, UINT32_MAX}) {
            check(!ltx_gpu_stage_workspace_create(gpu.get(), bad_rows, error, sizeof(error)));
            check(ltx_gpu_streaming_boundary_completed(gpu.get()));
            error[0] = 0;
        }
        check(ltx_gpu_batch_begin(gpu.get(), error, sizeof(error)));
        check(!ltx_gpu_stage_workspace_create(gpu.get(), 7u, error, sizeof(error)));
        check(ltx_gpu_batch_end(gpu.get(), error, sizeof(error)));
        error[0] = 0;
#ifdef TURBOCIDER_ENABLE_TEST_HOOKS
        for (unsigned failure = 1; failure <= 9u; ++failure) {
            ltx_gpu_stage_workspace_test_fail_at(failure);
            check(!ltx_gpu_stage_workspace_create(gpu.get(), 7u, error, sizeof(error)));
            check(ltx_gpu_streaming_boundary_completed(gpu.get()));
            ltx_gpu_stage_workspace_test_fail_at(0);
            error[0] = 0;
            auto *recovered = ltx_gpu_stage_workspace_create(gpu.get(), 7u, error, sizeof(error));
            check(recovered != nullptr);
            ltx_gpu_stage_workspace_free(recovered);
        }
        std::puts("PASS stage workspace partial allocation failures: nine sites, atomic attachment and recovery");
#endif

        constexpr uint32_t rows = 8u, dim = 4096u, heads = 32u, head_dim = 128u;
        auto host_x = values(size_t(rows) * dim, .7f);
        auto host_w = weights(size_t(dim) * dim);
        auto host_gate = values(size_t(heads) * dim, .001f);
        auto host_bias = values(dim, .02f);
        std::vector<uint16_t> host_norm(dim, bf16(1.f));
        std::vector<uint16_t> host_cos(size_t(rows) * dim / 2u), host_sin(host_cos.size());
        for (size_t i = 0; i < host_cos.size(); ++i) {
            float angle = float(i % 997u) * .03f;
            host_cos[i] = bf16(std::cos(angle));
            host_sin[i] = bf16(std::sin(angle));
        }
        std::vector<float> host_scale(dim, .0013f);
        auto x = tensor(gpu.get(), host_x.size() * 2u, host_x.data());
        auto w = tensor(gpu.get(), host_w.size(), host_w.data());
        auto scales = tensor(gpu.get(), host_scale.size() * 4u, host_scale.data());
        auto bias = tensor(gpu.get(), host_bias.size() * 2u, host_bias.data());
        auto norm = tensor(gpu.get(), host_norm.size() * 2u, host_norm.data());
        auto gate = tensor(gpu.get(), host_gate.size() * 2u, host_gate.data());
        auto cosine = tensor(gpu.get(), host_cos.size() * 2u, host_cos.data());
        auto sine = tensor(gpu.get(), host_sin.size() * 2u, host_sin.data());
        auto output = tensor(gpu.get(), host_x.size() * 2u);
        auto reference = tensor(gpu.get(), host_x.size() * 2u);
        auto self_attention = [&](ltx_gpu_buffer *out, const ltx_gpu_buffer *input) {
            return ltx_gpu_self_attention_int8_mps_bf16(gpu.get(), out, input,
                w.get(), scales.get(), bias.get(), w.get(), scales.get(), bias.get(),
                w.get(), scales.get(), bias.get(), norm.get(), norm.get(), gate.get(), nullptr,
                w.get(), scales.get(), bias.get(), cosine.get(), sine.get(),
                rows, heads, head_dim, 256u, 1e-6f, error, sizeof(error));
        };
        check(self_attention(reference.get(), x.get()));
        auto stage = std::unique_ptr<ltx_gpu_stage_workspace, decltype(&ltx_gpu_stage_workspace_free)>(
            ltx_gpu_stage_workspace_create(gpu.get(), rows, error, sizeof(error)),
            ltx_gpu_stage_workspace_free);
        check(stage != nullptr);
        check(ltx_gpu_stage_workspace_bytes(stage.get()) < (2u << 20));
        auto *hidden = ltx_gpu_stage_workspace_hidden(stage.get());
        check(hidden && ltx_gpu_buffer_bytes(hidden) >= size_t(rows) * 16384u * 2u);
        check(!ltx_gpu_stage_workspace_create(gpu.get(), rows, error, sizeof(error)));
        error[0] = 0;
        uint64_t before = ltx_gpu_stage_workspace_reuses(gpu.get());
        uint64_t fused_before = ltx_gpu_qk_norm_rope_dispatches(gpu.get());
        for (unsigned repeat = 0; repeat < 3u; ++repeat) {
            check(ltx_gpu_batch_begin(gpu.get(), error, sizeof(error)));
            check(self_attention(output.get(), x.get()));
            check(ltx_gpu_batch_end(gpu.get(), error, sizeof(error)));
            check(!memcmp(ltx_gpu_buffer_contents(reference.get()),
                          ltx_gpu_buffer_contents(output.get()), host_x.size() * 2u));
        }
        check(ltx_gpu_stage_workspace_reuses(gpu.get()) >= before + 24u);
        const char *fused_setting = std::getenv("TURBOCIDER_LTX_W8A8_FUSED_QK_NORM_ROPE");
        bool fused_qk = fused_setting ? !std::strcmp(fused_setting, "1") : rows >= 2048u;
        check(ltx_gpu_qk_norm_rope_dispatches(gpu.get()) == fused_before + (fused_qk ? 6u : 0u));
        /* Both graph layouts must coexist in one queue's cache. */
        setenv("TURBOCIDER_LTX_W8A8_FUSED_QK_NORM_ROPE", "0", 1);
        check(self_attention(output.get(), x.get()));
        check(!memcmp(ltx_gpu_buffer_contents(reference.get()),
                      ltx_gpu_buffer_contents(output.get()), host_x.size() * 2u));
        if (fused_setting) setenv("TURBOCIDER_LTX_W8A8_FUSED_QK_NORM_ROPE", fused_qk ? "1" : "0", 1);
        else unsetenv("TURBOCIDER_LTX_W8A8_FUSED_QK_NORM_ROPE");
        /* Pooling is scoped to video self-attention, then cleared before
         * audio/cross attention. It must retain the original preparation. */
        auto pooled_reference = tensor(gpu.get(), host_x.size() * 2u);
        check(ltx_gpu_set_self_kv_pool_geometry(gpu.get(), 2u, 2u, 2u, error, sizeof(error)));
        setenv("TURBOCIDER_LTX_W8A8_FUSED_QK_NORM_ROPE", "0", 1);
        check(self_attention(pooled_reference.get(), x.get()));
        setenv("TURBOCIDER_LTX_W8A8_FUSED_QK_NORM_ROPE", "1", 1);
        fused_before = ltx_gpu_qk_norm_rope_dispatches(gpu.get());
        check(self_attention(output.get(), x.get()));
        check(ltx_gpu_qk_norm_rope_dispatches(gpu.get()) == fused_before);
        check(!memcmp(ltx_gpu_buffer_contents(pooled_reference.get()),
                      ltx_gpu_buffer_contents(output.get()), host_x.size() * 2u));
        check(ltx_gpu_set_self_kv_pool_geometry(gpu.get(), 0u, 0u, 0u, error, sizeof(error)));
        if (fused_setting) setenv("TURBOCIDER_LTX_W8A8_FUSED_QK_NORM_ROPE", fused_qk ? "1" : "0", 1);
        else unsetenv("TURBOCIDER_LTX_W8A8_FUSED_QK_NORM_ROPE");
        check(!memcmp(ltx_gpu_buffer_contents(x.get()), host_x.data(), host_x.size() * 2u));

        /* The full FFN view overlaps every Q/K/V/core view. Passing it as an
         * external attention input must select independent temporary buffers. */
        std::memset(ltx_gpu_buffer_contents(hidden), 0, ltx_gpu_buffer_bytes(hidden));
        check(ltx_gpu_buffer_write(hidden, host_x.data(), host_x.size() * 2u, error, sizeof(error)));
        fused_before = ltx_gpu_qk_norm_rope_dispatches(gpu.get());
        check(self_attention(output.get(), hidden));
        check(ltx_gpu_qk_norm_rope_dispatches(gpu.get()) == fused_before);
        check(!memcmp(ltx_gpu_buffer_contents(reference.get()),
                      ltx_gpu_buffer_contents(output.get()), host_x.size() * 2u));
        check(!memcmp(ltx_gpu_buffer_contents(hidden), host_x.data(), host_x.size() * 2u));

        /* Alternate attention and an FFN using the same phase storage, under
         * an enclosing batch, to exercise real alias hazard dependencies. */
        constexpr uint32_t intermediate = 256u, outdim = 129u;
        auto first = weights(size_t(intermediate) * dim), second = weights(size_t(outdim) * intermediate);
        auto ws1 = tensor(gpu.get(), intermediate * 4u, host_scale.data());
        auto ws2 = tensor(gpu.get(), outdim * 4u, host_scale.data());
        auto fw1 = tensor(gpu.get(), first.size(), first.data());
        auto fw2 = tensor(gpu.get(), second.size(), second.data());
        auto ns_data = values(dim, .13f), nt_data = values(dim, .07f);
        auto ns = tensor(gpu.get(), dim * 2u, ns_data.data()), nt = tensor(gpu.get(), dim * 2u, nt_data.data());
        auto normalized = tensor(gpu.get(), host_x.size() * 2u);
        auto ffn_reference = tensor(gpu.get(), rows * outdim * 2u);
        auto ffn_actual = tensor(gpu.get(), rows * outdim * 2u);
        check(ltx_gpu_mlp_adaln_int8_convrot_mps_bf16(gpu.get(), ffn_reference.get(), x.get(),
            ns.get(), nt.get(), normalized.get(), 1e-6f, fw1.get(), ws1.get(), nullptr,
            fw2.get(), ws2.get(), nullptr, rows, dim, intermediate, outdim, 256u, error, sizeof(error)));
        for (unsigned repeat = 0; repeat < 3u; ++repeat) {
            check(ltx_gpu_batch_begin(gpu.get(), error, sizeof(error)));
            check(self_attention(output.get(), x.get()));
            check(ltx_gpu_mlp_adaln_int8_convrot_mps_bf16_with_scratch(gpu.get(), ffn_actual.get(), x.get(),
                hidden, ns.get(), nt.get(), normalized.get(), 1e-6f, fw1.get(), ws1.get(), nullptr,
                fw2.get(), ws2.get(), nullptr, rows, dim, intermediate, outdim, 256u, error, sizeof(error)));
            check(ltx_gpu_batch_end(gpu.get(), error, sizeof(error)));
            check(!memcmp(ltx_gpu_buffer_contents(ffn_reference.get()),
                          ltx_gpu_buffer_contents(ffn_actual.get()), rows * outdim * 2u));
        }
        /* Destroying an attached workspace drains a caller's pending batch. */
        check(ltx_gpu_batch_begin(gpu.get(), error, sizeof(error)));
        check(self_attention(output.get(), x.get()));
        stage.reset();
        check(ltx_gpu_streaming_boundary_completed(gpu.get()));
        before = ltx_gpu_stage_workspace_reuses(gpu.get());
        check(self_attention(output.get(), x.get()));
        check(ltx_gpu_stage_workspace_reuses(gpu.get()) == before);
        check(!memcmp(ltx_gpu_buffer_contents(reference.get()),
                      ltx_gpu_buffer_contents(output.get()), host_x.size() * 2u));

        auto other = std::unique_ptr<ltx_gpu, decltype(&ltx_gpu_free)>(
            ltx_gpu_create(argv[1], error, sizeof(error)), ltx_gpu_free);
        check(other != nullptr);
        stage.reset(ltx_gpu_stage_workspace_create(other.get(), 7u, error, sizeof(error)));
        check(stage != nullptr);
        other.reset();
        stage.reset();
        if (fused_qk) std::puts("PASS fused Q/K RMSNorm/RoPE/layout: exact native baseline, nontrivial angles, layout cache alternation, alias fallback and dependent batches");
        std::puts("PASS stage workspace: exact QKV/core/projection reuse, aliased-input fallback, dependent attention/FFN batches, pending-batch drain, nested rejection and GPU-before-workspace teardown");
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "%s\n", e.what());
        return 1;
    }}
}
