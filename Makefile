.DEFAULT_GOAL := help
.PHONY: help build package test test-model doctor
help:
	@echo 'TurboCider — native FLUX inference system'
	@echo 'MLX_ROOT=/path/to/mlx make build    Build engine, CLI, App and Swift tests'
	@echo 'MLX_ROOT=/path/to/mlx make package  Build and create dist/TurboCider.app + dist/cli'
	@echo 'make test                         Verify repository boundaries and request contracts'
	@echo 'make test-model MODEL=/path/to/FLUX.2-klein-4B OUTPUT=/tmp/new-tc-validation'
	@echo 'make doctor                       Inspect this Mac and native dependencies'
build:
	@tools/native/build.sh
package: build
	@tools/native/package.sh
test:
	@python3 tests/repository/test_layout.py
	@python3 tests/native/test_contract.py
test-model:
	@test -n "$(MODEL)" -a -n "$(OUTPUT)" || (echo 'MODEL and OUTPUT are required'; exit 1)
	@build/native/turbocider self-test
	@build/native/turbocider-lifecycle-test "$(MODEL)" "$(OUTPUT)/lifecycle"
	@python3 tests/native/test_service.py --model "$(MODEL)" --output "$(OUTPUT)/service"
doctor:
	@build/native/turbocider doctor
