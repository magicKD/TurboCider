.DEFAULT_GOAL := help
# Prefer the project environment so tests use the same dependencies as the build.
LOCAL_PYTHON := $(firstword $(wildcard .venv/bin/python3 .deps/bin/python3.11))
PYTHON ?= $(if $(LOCAL_PYTHON),$(LOCAL_PYTHON),python3.11)
export PATH := $(CURDIR)/.venv/bin:$(CURDIR)/.deps/bin:$(PATH)
.PHONY: help setup build build-app build-vision-quality package test test-app test-model doctor h3-quant-cache test-library test-api test-video-preview
.PHONY: test-streaming-host test-streaming-contract test-streaming-metal test-streaming-campaign test-streaming-catalog-builder test-streaming-source-identity test-streaming-source-lease test-streaming-audit test-streaming-pager test-ltx-streaming-lifecycle test-ltx-streaming-lifecycle-faults test-process-tree-sampler
help:
	@echo 'TurboCider — native multimodal inference system'
	@echo 'MLX_ROOT=/path/to/mlx make build    Build engine, CLI, App and Swift tests'
	@echo 'MLX_ROOT=/path/to/mlx make package  Build and create dist/TurboCider.app + dist/cli'
	@echo 'make setup                       Install pinned, TurboCider-owned dependencies'
	@echo 'make build                       Build engine, CLI, App and Swift tests'
	@echo 'make package                      Build and create dist/TurboCider.app + dist/cli'
	@echo 'make build-app                    Rebuild Swift UI after an engine build'
	@echo 'make build-vision-quality         Build optional public-Vision quality helper'
	@echo 'make test-app                     Run App behavior tests (macOS clipboard access)'
	@echo 'make test-library                 Verify model library using tiny loopback downloads'
	@echo 'make test                         Verify repository boundaries and request contracts'
	@echo 'make test-streaming-host           Verify layout/executor/LTX metadata with synthetic fixtures'
	@echo 'make test-streaming-contract       Verify streaming API/snapshot contracts (requires native build)'
	@echo 'make test-streaming-metal          Verify synthetic GPU slots (requires Metal access)'
	@echo 'make test-streaming-campaign       Verify CPU-only ABBA campaign runner/verifier'
	@echo 'make test-streaming-catalog-builder Verify immutable catalog builder contracts'
	@echo 'make test-streaming-source-identity Verify source/build provenance capture'
	@echo 'make test-streaming-source-lease    Verify fd lease, mutation and value snapshot contracts'
	@echo 'make test-streaming-audit          Verify audit-only counters and release symbol isolation'
	@echo 'make test-streaming-pager          Verify sparse MLX resident/slot pager failure contracts'
	@echo 'make test-ltx-streaming-lifecycle MODEL=/path OUTPUT=/path  Opt-in real LTX exact lifecycle test'
	@echo 'make test-ltx-streaming-lifecycle-faults MODEL=/path OUTPUT=/path  Require a test-hook build and unsafe-cleanup matrix'
	@echo 'make test-model MODEL=/path/to/FLUX.2-klein-4B OUTPUT=/tmp/new-tc-validation'
	@echo 'make doctor                       Inspect this Mac and native dependencies'
	@echo 'make h3-quant-cache MODEL=/path/to/transformer OUTPUT=/path/to/cache'
setup:
	@"$(PYTHON)" tools/setup_dependencies.py
build-app:
	@tools/native/build_app.sh
build:
	@tools/native/build.sh
build-vision-quality:
	@tools/native/build_vision_feature_distance.sh
package: build
	@tools/native/package.sh
test:
	@"$(PYTHON)" tests/repository/test_layout.py
	@"$(PYTHON)" tests/repository/test_independence.py
	@"$(PYTHON)" tests/repository/test_cpp_boundaries.py
	@"$(PYTHON)" tests/native/test_hash_small_stack.py
	@"$(PYTHON)" tests/native/test_contract.py
	@$(MAKE) test-streaming-host
	@$(MAKE) test-streaming-contract
	@$(MAKE) test-streaming-campaign
	@$(MAKE) test-streaming-catalog-builder
	@$(MAKE) test-streaming-source-identity
	@$(MAKE) test-streaming-audit
	@$(MAKE) test-streaming-pager
	@"$(PYTHON)" -B tests/native/test_streaming_metal.py
	@"$(PYTHON)" -B tests/native/test_z_image_sharded_checkpoint.py
	@"$(PYTHON)" -B tests/native/test_z_image_weight_stream.py
	@"$(PYTHON)" -B tests/native/test_coreml_lora.py
	@"$(PYTHON)" -B tests/native/test_llada_reference.py
	@"$(PYTHON)" -B tests/native/test_quality_gate.py
	@"$(PYTHON)" -B tests/native/test_video_quality_gate.py
	@"$(PYTHON)" -B tests/native/test_video_timing.py
	@"$(PYTHON)" -B tests/native/test_wan_benchmark.py
	@"$(PYTHON)" -B tests/native/test_native_gguf.py
	@"$(PYTHON)" -B tests/native/test_nvfp4.py
	@"$(PYTHON)" -B tests/native/test_h3_streaming_policy.py
	@"$(PYTHON)" -B tests/native/test_memory_accounting.py
	@"$(PYTHON)" -B tests/native/test_memory_manifest.py
	@"$(PYTHON)" -B tests/native/test_memory_schedule.py
	@"$(PYTHON)" -B tests/native/test_memory_schedule_adapter.py
	@"$(PYTHON)" -B tests/native/test_memory_plan_compiler.py
	@"$(PYTHON)" -B tests/native/test_memory_scheduler.py
	@"$(PYTHON)" -B tests/native/test_memory_watchdog.py
	@"$(PYTHON)" -B tests/native/test_memory_trace.py
	@"$(PYTHON)" -B tests/native/test_h3_schedule_memory.py
	@"$(PYTHON)" -B tests/native/test_memory_execution.py
	@"$(PYTHON)" -B tests/native/test_memory_probe.py
	@"$(PYTHON)" -B tests/native/test_h3_gpu_memory_hooks.py
	@"$(PYTHON)" -B tests/native/test_ltx_gpu_memory_hooks.py
	@"$(PYTHON)" -B tests/native/test_h3_quant_cache.py
	@"$(PYTHON)" -B tests/native/test_h3_mlx_source_contract.py
	@"$(PYTHON)" -B tests/native/test_h3_mlx_geometry.py
	@"$(PYTHON)" -B tests/native/test_h3_mlx_cache.py
	@"$(PYTHON)" -B tests/native/test_h3_mlx_prepare.py
	@"$(PYTHON)" -B tests/native/test_h3_mlx_modelscope.py
	@"$(PYTHON)" -B tests/native/test_vdn_modelscope.py
	@"$(PYTHON)" -B tests/native/test_vdn_mlx_solve.py
	@"$(PYTHON)" tests/native/test_inventory.py
# No real model weights, full inference, or system memory pressure in these
# focused targets. GPU tests may report SKIP when Metal access is unavailable.
test-streaming-host:
	@"$(PYTHON)" -B tests/native/test_streaming_layout.py
	@"$(PYTHON)" -B tests/native/test_streaming_actual_receipt.py
	@$(MAKE) test-streaming-source-lease
	@"$(PYTHON)" -B tests/native/test_streaming_preset_resolver.py
	@"$(PYTHON)" -B tests/native/test_ltx_streaming_layout.py
	@"$(PYTHON)" -B tests/native/test_ltx_streaming_descriptor.py
	@"$(PYTHON)" -B tests/native/test_h3_streaming_descriptor.py
	@"$(PYTHON)" -B tests/native/test_z_image_streaming_descriptor.py
	@"$(PYTHON)" -B tests/native/test_flux_streaming_descriptor.py
test-process-tree-sampler:
	@"$(PYTHON)" -B tests/native/test_process_tree_sampler.py
test-streaming-contract:
	@"$(PYTHON)" -B tests/native/test_ltx_streaming_snapshot.py
	@"$(PYTHON)" -B tests/native/test_ltx_finalizer_envelope.py
	@"$(PYTHON)" -B tests/native/test_streaming_contract.py
	@"$(PYTHON)" -B tests/native/test_streaming_test_catalog.py
	@"$(PYTHON)" -B tests/native/test_ltx_candidate_streaming_gate.py
	@"$(PYTHON)" -B tests/native/test_ltx_public_streaming.py
	@"$(PYTHON)" -B tests/native/test_h3_candidate_streaming_gate.py
	@"$(PYTHON)" -B tests/native/test_h3_public_streaming.py
	@"$(PYTHON)" -B tests/native/test_z_image_candidate_streaming_gate.py
	@"$(PYTHON)" -B tests/native/test_z_image_public_streaming.py
	@"$(PYTHON)" -B tests/native/test_flux_candidate_streaming_gate.py
	@"$(PYTHON)" -B tests/native/test_flux_public_streaming.py
test-streaming-pager:
	@"$(PYTHON)" -B tests/native/test_mlx_weight_pager.py
	@"$(PYTHON)" -B tests/native/test_mlx_weights_lease.py
test-streaming-metal:
	@$(MAKE) test-streaming-pager
	@"$(PYTHON)" -B tests/native/test_streaming_metal.py
	@"$(PYTHON)" -B tests/native/test_ltx_streaming_layout.py --metal
test-streaming-campaign:
	@"$(PYTHON)" -B tests/native/test_streaming_campaign_verifier.py
test-streaming-catalog-builder:
	@"$(PYTHON)" -B tests/native/test_streaming_catalog_builder.py
	@"$(PYTHON)" -B tests/native/test_prepare_streaming_release_policies.py
test-streaming-source-identity:
	@"$(PYTHON)" -B tests/native/test_streaming_source_identity.py
test-streaming-source-lease:
	@"$(PYTHON)" -B tests/native/test_streaming_source_lease.py
test-streaming-audit:
	@"$(PYTHON)" -B tests/native/test_streaming_audit.py
test-ltx-streaming-lifecycle:
	@test -n "$(MODEL)" -a -n "$(OUTPUT)" || (echo 'MODEL=/path/to/LTX-2.5 and OUTPUT=/path/to/results are required'; exit 1)
	@"$(PYTHON)" -B tests/native/test_ltx_candidate_streaming_lifecycle.py \
		--library build/native/libturbocider.dylib --model "$(MODEL)" \
		--cache "$(OUTPUT)/cache" --output "$(OUTPUT)/lifecycle"
test-ltx-streaming-lifecycle-faults:
	@test -n "$(MODEL)" -a -n "$(OUTPUT)" || (echo 'MODEL=/path/to/LTX-2.5 and OUTPUT=/path/to/results are required'; exit 1)
	@"$(PYTHON)" -B tests/native/test_ltx_candidate_streaming_lifecycle.py \
		--library build/native/libturbocider.dylib --model "$(MODEL)" \
		--cache "$(OUTPUT)/cache" --output "$(OUTPUT)/lifecycle" \
		--require-test-hooks
test-app:
	@build/native/turbocider-ane-library-tests
	@build/native/turbocider-studio-tests
	@build/native/turbocider-model-library-tests
	@build/native/turbocider-library-store-tests
	@build/native/turbocider-run-insights-tests
	@build/native/turbocider-tensor-cache-tests
test-library:
	@build/native/turbocider-library-store-tests
	@build/native/turbocider-installation-tests
	@build/native/turbocider-library-tool-tests
	@"$(PYTHON)" tests/native/test_model_library_download.py
test-video-preview:
	@test -n "$(VIDEO)" || (echo 'VIDEO=/path/to/generated.mp4 is required'; exit 1)
	@build/native/turbocider-video-preview-tests "$(VIDEO)"
test-api:
	@build/native/turbocider-local-api-tests
	@"$(PYTHON)" tests/native/test_service_lifecycle.py
test-model:
	@test -n "$(MODEL)" -a -n "$(OUTPUT)" || (echo 'MODEL and OUTPUT are required'; exit 1)
	@build/native/turbocider-studio-model-tests "$(MODEL)" "$(OUTPUT)/studio"
	@build/native/turbocider self-test
	@build/native/turbocider-lifecycle-test "$(MODEL)" "$(OUTPUT)/lifecycle"
	@"$(PYTHON)" tests/native/test_service.py --model "$(MODEL)" --output "$(OUTPUT)/service"
doctor:
	@build/native/turbocider doctor
h3-quant-cache: build
	@test -n "$(MODEL)" -a -n "$(OUTPUT)" || (echo 'MODEL and OUTPUT are required'; exit 1)
	@build/native/h3-quantize-stream-cache --transformer "$(MODEL)" --shader build/native/h3_shaders.metal --output "$(OUTPUT)"
