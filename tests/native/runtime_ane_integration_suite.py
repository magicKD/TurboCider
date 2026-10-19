"""Bounded CPU-only regression suite for the integrated runtime ANE work.

Keep actual GPU lifecycle tests and opt-in checkpoint/device probes outside
this list. All modules are explicit so a new full-model test cannot silently
enter the default no-inference contract target through a filename wildcard.
"""
import unittest


HOST_MODULES = (
    "test_ane_fixed_program", "test_ane_multi_function", "test_ane_procedure",
    "test_ane_w8_activation", "test_ane_w8_functions", "test_ane_w8_functions_oracle",
    "test_ane_w8_projection",
    "test_qwen21_base_runtime_lora_options", "test_qwen21_checkpoint_recipe",
    "test_qwen21_compiled_complement_options", "test_qwen21_edit_qkv_step0",
    "test_qwen21_gpu_dual_options", "test_qwen21_private_qkv_options",
    "test_qwen21_runtime_cache_options",
    "test_qwen21_runtime_functions", "test_qwen21_runtime_prefix_options",
    "test_qwen_async_step_cancel_screen", "test_qwen_async_step_pair_audit",
    "test_qwen_async_step_pair_screen", "test_qwen_base_activation",
    "test_qwen_base_activation_operator", "test_qwen_base_block_observer_options",
    "test_qwen_base_compiled_complement", "test_qwen_base_convrot_factorized",
    "test_qwen_base_down", "test_qwen_base_dual_gup", "test_qwen_base_expanded",
    "test_qwen_base_fixed_program", "test_qwen_base_fp16_qkv", "test_qwen_base_frontier",
    "test_qwen_base_gpu_dual_request_audit", "test_qwen_base_gpu_dual_runtime",
    "test_qwen_base_handoff", "test_qwen_base_hidden_quant", "test_qwen_base_joint_profile",
    "test_qwen_base_mixed_hidden", "test_qwen_base_neighbor_share",
    "test_qwen_base_profile_audits", "test_qwen_base_profiling", "test_qwen_base_projection",
    "test_qwen_base_qkv_lookahead", "test_qwen_base_qkv_profile", "test_qwen_base_qkv_stage",
    "test_qwen_base_rotation", "test_qwen_base_runtime_prefix_profile",
    "test_qwen_base_runtime_screen", "test_qwen_base_share_quality",
    "test_qwen_base_swiglu_block_screen", "test_qwen_base_text_prefix",
    "test_qwen_base_vpipe_compare", "test_qwen_cache_hybrid_screen",
    "test_qwen_compiled_complement_request", "test_qwen_compiled_complement_request_audit",
    "test_qwen_corrected_qkv_request_screen", "test_qwen_edit_step0_profile",
    "test_qwen_function_request_audit", "test_qwen_function_requests",
    "test_qwen_lora_swiglu_block_audit", "test_qwen_lora_swiglu_block_screen",
    "test_qwen_lora_swiglu_native_source", "test_qwen_lora_swiglu_request_audit",
    "test_qwen_lora_swiglu_request_screen", "test_qwen_prefix_cache_screen",
    "test_qwen_qkv_lora_block_screen", "test_qwen_qkv_numeric_screen",
    "test_qwen_qkv_prefix_composition_screen", "test_qwen_qkv_private_combo_screen",
    "test_qwen_runtime_cache_audit", "test_qwen_runtime_cache_profile",
    "test_qwen_sampler_qkv_composition_screen", "test_qwen_static_lora_block_screen",
    "test_qwen_student_graph_screen", "test_qwen_student_prefix_screen",
    "test_qwen_student_rank_report", "test_qwen_student_sampler_screen",
    "test_qwen_tokenizer_reuse_screen", "test_qwen_vae_precision_screen",
    "test_qwen_w8_functions_real", "test_qwen_w8_functions_real_audit",
    "test_runtime_image_collect",
)


def load_tests(loader, tests, pattern):
    return loader.loadTestsFromNames(HOST_MODULES)


if __name__ == "__main__":
    unittest.main(verbosity=2)
