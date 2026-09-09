.DEFAULT_GOAL := help
.PHONY: help setup build build-app package test test-app test-library test-api test-model doctor
help:
	@echo 'TurboCider — native multimodal inference system'
	@echo 'MLX_ROOT=/path/to/mlx make build    Build engine, CLI, App and Swift tests'
	@echo 'MLX_ROOT=/path/to/mlx make package  Build and create dist/TurboCider.app + dist/cli'
	@echo 'make setup                       Install pinned, TurboCider-owned dependencies'
	@echo 'make build                       Build engine, CLI, App and Swift tests'
	@echo 'make package                      Build and create dist/TurboCider.app + dist/cli'
	@echo 'make build-app                    Rebuild Swift UI after an engine build'
	@echo 'make test-app                     Run App behavior tests (macOS clipboard access)'
	@echo 'make test-library                 Verify model library using tiny loopback downloads'
	@echo 'make test                         Verify repository boundaries and request contracts'
	@echo 'make test-model MODEL=/path/to/FLUX.2-klein-4B OUTPUT=/tmp/new-tc-validation'
	@echo 'make doctor                       Inspect this Mac and native dependencies'
setup:
	@python3.11 tools/setup_dependencies.py
build-app:
	@tools/native/build_app.sh
build:
	@tools/native/build.sh
package: build
	@tools/native/package.sh
test:
	@python3 tests/repository/test_layout.py
	@python3 tests/repository/test_independence.py
	@python3 tests/repository/test_cpp_boundaries.py
	@python3 tests/native/test_hash_small_stack.py
	@python3 tests/native/test_contract.py
	@python3 -B tests/native/test_z_image_sharded_checkpoint.py
	@python3 -B tests/native/test_coreml_lora.py
	@python3 tests/native/test_inventory.py
test-app:
	@build/native/turbocider-studio-tests
	@build/native/turbocider-model-library-tests
	@build/native/turbocider-library-store-tests
	@build/native/turbocider-run-insights-tests
	@build/native/turbocider-tensor-cache-tests
test-library:
	@build/native/turbocider-library-store-tests
	@build/native/turbocider-installation-tests
	@build/native/turbocider-library-tool-tests
	@python3 tests/native/test_model_library_download.py
test-api:
	@build/native/turbocider-local-api-tests
	@python3 tests/native/test_service_lifecycle.py
test-model:
	@test -n "$(MODEL)" -a -n "$(OUTPUT)" || (echo 'MODEL and OUTPUT are required'; exit 1)
	@build/native/turbocider-studio-model-tests "$(MODEL)" "$(OUTPUT)/studio"
	@build/native/turbocider self-test
	@build/native/turbocider-lifecycle-test "$(MODEL)" "$(OUTPUT)/lifecycle"
	@python3 tests/native/test_service.py --model "$(MODEL)" --output "$(OUTPUT)/service"
doctor:
	@build/native/turbocider doctor
