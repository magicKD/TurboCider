import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


class H3MLXSourceContractTests(unittest.TestCase):
    def test_int6_affine_geometry_is_explicit(self):
        header = (ROOT / "native/components/weights/affine.hpp").read_text()
        source = (ROOT / "native/components/weights/affine.cpp").read_text()
        self.assertIn("int logical_input_channels = 0", header)
        self.assertIn("bits == 6", source)
        self.assertIn("int64_t(input_channels_) * bits", source)
        self.assertIn("32 % bits_ == 0", source)
        self.assertNotIn("input_channels_ = packed_.shape(1) * (32 / bits)", source)

    def test_h3_loader_requires_fastvideo_int6_contract(self):
        source = (ROOT / "native/platform/apple/h3_mlx_checkpoint.mm").read_text()
        build = (ROOT / "tools/native/build.sh").read_text()
        for token in [
            '"mlx_h3_dit.json"',
            '"mlx_h3_dit.safetensors"',
            '@"format_version", 1',
            '@"bits", 6',
            '@"group_size", 64',
            'timesteps.count == expected_ladder.size()',
            'adaln_timestep_union(expected_steps)',
            '"__adaln_cache."',
            'affine_dq_gemm_min_rows_ = 768',
            'mx::float32',
        ]:
            self.assertIn(token, source)
        self.assertIn("native/platform/apple/h3_mlx_checkpoint.mm", build)
        self.assertIn("native/models/h3_mlx/dit.cpp", build)
        self.assertIn("native/platform/apple/h3_mlx_prompt_cache.mm", build)
        self.assertIn("h3_mlx_tensor_probe.cpp", build)
        self.assertIn("h3_mlx_tokenizer_probe.cpp", build)
        self.assertIn("h3_mlx_conditioner_probe.cpp", build)
        self.assertIn("h3_mlx_encoder_benchmark_probe.cpp", build)
        self.assertIn("h3_mlx_mlp_hybrid_probe.cpp", build)
        self.assertIn("TURBOCIDER_BUILD_EXPERIMENTAL_PROBES", build)
        self.assertIn("h3_mlx_prompt_cache_probe.cpp", build)
        self.assertIn("h3_mlx_vsa_probe.cpp", build)
        self.assertIn("h3_mlx_vsa_pipeline_probe.cpp", build)
        self.assertIn("h3_mlx_vsa_e2e_probe.cpp", build)
        self.assertIn("h3_mlx_vdn_solve_probe.cpp", build)
        self.assertIn("h3_mlx_vdn_e2e_probe.cpp", build)

    def test_vdn_schedule_is_checkpoint_bound_not_hardcoded_to_fast_h3(self):
        geometry = (ROOT / "native/models/h3_mlx/geometry.cpp").read_text()
        header = (ROOT / "native/models/h3_mlx/geometry.hpp").read_text()
        loader = (ROOT / "native/platform/apple/h3_mlx_checkpoint.mm").read_text()
        pipeline = (ROOT / "native/models/h3_mlx/pipeline.cpp").read_text()
        self.assertIn("adaln_timestep_union(int steps)", header)
        self.assertIn("return adaln_timestep_union(4);", geometry)
        self.assertIn('profile == "minimax-h3-vdn"', loader)
        self.assertIn("expected_steps = vdn_profile ? 6 : 4", loader)
        self.assertIn("adaln_timestep_union(expected_steps)", loader)
        self.assertIn("options.steps == checkpoint_.identity().steps", pipeline)
        self.assertNotIn('options.steps == 4, "FastH3 MLX', pipeline)

    def test_results_exposes_fastvideo_int6_graph(self):
        source = (ROOT / "native/platform/apple/results.mm").read_text()
        self.assertIn('fasth3_int6_qmm', source)
        self.assertIn('fasth3_int6_vsa', source)
        self.assertIn('int6_g64_bf16_activation', source)

    def test_qwen3_vl_encoder_hybrid_is_manifest_bound_and_streamed(self):
        conditioner = (ROOT / "native/models/h3_mlx/conditioner.cpp").read_text()
        header = (ROOT / "native/models/h3_mlx/conditioner.hpp").read_text()
        session = (ROOT / "native/platform/apple/h3_mlx_session.mm").read_text()
        shards = (ROOT / "native/platform/apple/h3_mlx_shards.mm").read_text()
        probe = (ROOT / "tools/native/h3_mlx_conditioner_probe.cpp").read_text()
        results = (ROOT / "native/platform/apple/results.mm").read_text()
        resources = (ROOT / "native/backends/coreml_resources.mm").read_text()
        for token in [
            "hybrid_manifest",
            "hybrid_session",
            "conditioner_checkpoint(component_root_)",
            "ane_mlp_end",
            "TURBOCIDER_H3_QWEN3_HYBRID_VALIDATE",
            "H3 Qwen3 hybrid MLP quality gate failed",
        ]:
            self.assertIn(token, conditioner + header)
        self.assertIn("load_slice", shards)
        self.assertIn("STRIDED_READ_SLAB_BYTES", shards)
        self.assertIn(
            "source_row_bytes - source_row_bytes / 2u", shards,
        )
        self.assertIn("scratch.data() + local * source_row_bytes", shards)
        self.assertGreaterEqual(shards.count("@autoreleasepool"), 3)
        self.assertIn("select_conditioner(request)", session)
        self.assertIn("encoder_ane_manifest.empty()", session)
        self.assertIn("qwen3_vl_encoder_mlp_complement", results)
        self.assertIn("--encoder-ane-manifest", probe)
        self.assertTrue((ROOT / "tools/native/h3_mlx_mlp_hybrid_probe.cpp").is_file())
        self.assertIn('model_id=="minimax-h3-fasth3-mlx-int6"', resources)
        self.assertIn("ane_mlp_limit=h3?25599", resources)

    def test_qwen3_vl_encoder_sdpa_candidate_is_opt_in_and_validated(self):
        source = (ROOT / "native/models/h3_mlx/conditioner.cpp").read_text()
        for token in [
            'TURBOCIDER_H3_FUSED_SDPA',
            'TURBOCIDER_H3_DISABLE_FUSED_SDPA',
            'TURBOCIDER_H3_FUSED_SDPA_MIN_TOKENS',
            'TURBOCIDER_H3_FUSED_SDPA_VALIDATE',
            'TURBOCIDER_H3_FUSED_SDPA_MAX_RELATIVE_L2',
            'TURBOCIDER_H3_FUSED_SDPA_MIN_COSINE',
            'TURBOCIDER_H3_FUSED_SDPA_MAX_RELATIVE_ABS',
            'tc::attend(query_heads, key_heads, value_heads',
            'H3 fused SDPA quality gate failed',
        ]:
            self.assertIn(token, source)
        self.assertIn('"causal"', source)

    def test_vsa_profile_is_capability_gated_and_observable(self):
        module = (ROOT / "native/models/h3_mlx_module.cpp").read_text()
        checkpoint = (ROOT / "native/platform/apple/h3_mlx_checkpoint.mm").read_text()
        session = (ROOT / "native/platform/apple/h3_mlx_session.mm").read_text()
        build = (ROOT / "tools/native/build.sh").read_text()
        for token in [
            'minimax-h3-fasth3-mlx-int6-vsa',
            'VSA profile requires parameters.vsa=true',
            'FastH3 VSA sparsity must be in [0, 1)',
            'FastH3 VSA tile_size must be 64 or 256',
            'FastH3 VSA dense layer indices must be unique',
        ]:
            self.assertIn(token, module)
        for token in [
            '@"num_gate_matrices"',
            'identity_.vsa_capable',
            'identity_.vsa_gate_matrices == config_.num_layers',
            'vsa_gates.size() == size_t(identity_.vsa_gate_matrices)',
        ]:
            self.assertIn(token, checkpoint)
        for token in [
            'FastVideo/FastVideo-FastH3-4-step-Preview-v1-VSA-DataFree',
            'FastH3 MLX VSA INT6 checkpoint is missing',
            'root / "vsa-int6" / "int6"',
            '@"configured_sparsity"',
            '@"achieved_sparsity"',
            '@"dense_fallback_reason"',
        ]:
            self.assertIn(token, session)
        for path in [
            'native/models/h3_mlx/vsa.cpp',
            'native/models/h3_mlx/vsa_attention.cpp',
        ]:
            self.assertIn(path, build)

    def test_vdn_profile_is_independent_and_fail_closed(self):
        module = (ROOT / "native/models/h3_mlx_module.cpp").read_text()
        registry = (ROOT / "native/models/registry.cpp").read_text()
        results = (ROOT / "native/platform/apple/results.mm").read_text()
        for token in [
            'minimax-h3-vdn',
            'VDN H3 stage-DMD requires exactly six steps',
            'VDN H3 uses the manifest-bound stage-DMD Turbo adapter',
            'd.executable = true',
            'create_h3_mlx_vdn',
            'window softmax + bidirectional VDN solve',
        ]:
            self.assertIn(token, module)
        self.assertIn('h3_mlx_vdn_module()', registry)
        self.assertIn('h3_vdn_int6_window_delta', results)
        self.assertIn('int6_g64_base+bf16_vdn+fp32_solve', results)
        self.assertIn('vdn_ ? @"int6_g64_base+bf16_vdn+fp32_solve"',
                      (ROOT / "native/platform/apple/h3_mlx_session.mm").read_text())

    def test_vdn_transition_scales_inverse_rows(self):
        source = (ROOT / "native/models/h3_mlx/vdn_mlx.cpp").read_text()
        self.assertIn("mx::expand_dims(alpha, 3) * inv", source)
        self.assertNotIn("mx::expand_dims(alpha, 2) * inv", source)

    def test_vdn_experimental_memory_and_gemm_knobs_are_opt_in(self):
        pipeline = (ROOT / "native/models/h3_mlx/pipeline.cpp").read_text()
        branch = (ROOT / "native/models/h3_mlx/vdn_mlx.cpp").read_text()
        self.assertIn('"TURBOCIDER_VDN_DQ_GEMM_MIN_ROWS"', pipeline)
        self.assertIn('"TURBOCIDER_VDN_COMPACT_LINEAR_OUTPUT"', branch)
        self.assertIn("options.affine_dq_gemm_min_rows", pipeline)
        self.assertIn("auto zeros = mx::zeros", branch)

    def test_vdn_window_indices_are_cached_with_diagnostic_opt_out(self):
        branch = (ROOT / "native/models/h3_mlx/vdn_mlx.cpp").read_text()
        self.assertIn("VDNWindowIndexCache", branch)
        self.assertIn("static thread_local std::optional<VDNWindowIndexCache> cache",
                      branch)
        self.assertIn("TURBOCIDER_VDN_DISABLE_WINDOW_INDEX_CACHE", branch)
        self.assertIn("if (disable_cache) cache.reset();", branch)

    def test_vdn_fused_qkv_is_explicit_and_observable(self):
        dit = (ROOT / "native/models/h3_mlx/dit.cpp").read_text()
        pipeline = (ROOT / "native/models/h3_mlx/pipeline.cpp").read_text()
        session = (ROOT / "native/platform/apple/h3_mlx_session.mm").read_text()
        header = (ROOT / "native/models/h3_mlx/pipeline.hpp").read_text()
        self.assertIn("weights_.experimental_fused_qkv()", dit)
        self.assertIn("TURBOCIDER_VDN_EXPERIMENTAL_FUSED_QKV", pipeline)
        self.assertIn("experimental_fused_qkv = false", header)
        self.assertIn('@"experimental_fused_qkv"', session)

    def test_session_removes_video_only_file_after_successful_mux(self):
        source = (ROOT / "native/platform/apple/h3_mlx_session.mm").read_text()
        mux = source.index("mux_video_with_audio(video_only")
        remove = source.index("std::filesystem::remove(video_only", mux)
        catch = source.index("} catch (...) {", mux)
        self.assertLess(remove, catch)

    def test_session_reports_per_phase_mlx_peaks(self):
        source = (ROOT / "native/platform/apple/h3_mlx_session.mm").read_text()
        for key in [
            '@"condition_peak_bytes"',
            '@"denoise_peak_bytes"',
            '@"video_decode_peak_bytes"',
            '@"audio_decode_peak_bytes"',
            "mx::get_active_memory()",
        ]:
            self.assertIn(key, source)

    def test_session_accepts_explicit_fastvideo_noise_fixture(self):
        source = (ROOT / "native/platform/apple/h3_mlx_session.mm").read_text()
        for token in [
            '"video_noise"',
            '"audio_noise"',
            'load_noise_fixture',
            '@"noise_fixture"',
        ]:
            self.assertIn(token, source)

    def test_e2e_parity_can_dump_conditioning_and_final_latents(self):
        session = (ROOT / "native/platform/apple/h3_mlx_session.mm").read_text()
        runner = (ROOT / "tools/h3/run_fastvideo_fasth3_noise_e2e.py").read_text()
        for token in [
            '"h3_mlx_e2e.safetensors"',
            '"conditioning"',
            '"token_tags"',
            '"video_rows"',
            '"audio_rows"',
        ]:
            self.assertIn(token, session)
            self.assertIn(token, runner)
        self.assertIn('--dump-tensors', runner)
        self.assertIn('--conditioning-input', runner)
        self.assertIn('--stop-after-denoise', runner)
        for token in [
            '--vsa-sparsity', '--vsa-tile-size', '--vsa-prefix-mode',
            '--vsa-dense-first-n-steps', '--vsa-dense-layers', '--vsa-impl',
            'MiniMaxH3VSAConfig', 'dit.prepare_vsa_geometry(layout)',
        ]:
            self.assertIn(token, runner)

    def test_dit_uses_cached_adaln_and_per_block_materialization(self):
        source = (ROOT / "native/models/h3_mlx/dit.cpp").read_text()
        for token in [
            '"__adaln_cache.block."',
            '"__adaln_cache.norm_out_shift"',
            'mx::fast::scaled_dot_product_attention',
            'rows >= affine_dq_gemm_min_rows_',
            'mx::eval(packed)',
            'scheduler_step(',
        ]:
            if token == 'rows >= affine_dq_gemm_min_rows_':
                self.assertIn(token, (ROOT / "native/platform/apple/h3_mlx_checkpoint.mm").read_text())
            else:
                self.assertIn(token, source)

    def test_video_vae_preserves_sequence_major_batch_and_python_modulo(self):
        source = (ROOT / "native/models/h3_mlx/video_vae.cpp").read_text()
        self.assertIn("const int batch = input.shape(1);", source)
        self.assertIn(
            "{sequence, batch, config_.num_heads * config_.head_dim}",
            source,
        )
        self.assertIn(
            "positive_mod(-config_.token_drop, tokens_chunk)",
            source,
        )
        self.assertIn(
            "positive_mod(-num_tokens, tokens_chunk)",
            source,
        )
        self.assertIn(
            "positive_mod(-config_.clip_length, config_.temporal_ratio)",
            source,
        )


if __name__ == "__main__":
    unittest.main()
