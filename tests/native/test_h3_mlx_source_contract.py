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
            'timesteps.count == 8',
            'four_step_adaln_union()',
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
        self.assertIn("h3_mlx_prompt_cache_probe.cpp", build)
        self.assertIn("h3_mlx_vsa_probe.cpp", build)
        self.assertIn("h3_mlx_vsa_pipeline_probe.cpp", build)
        self.assertIn("h3_mlx_vsa_e2e_probe.cpp", build)

    def test_results_exposes_fastvideo_int6_graph(self):
        source = (ROOT / "native/platform/apple/results.mm").read_text()
        self.assertIn('fasth3_int6_qmm', source)
        self.assertIn('fasth3_int6_vsa', source)
        self.assertIn('int6_g64_bf16_activation', source)

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
