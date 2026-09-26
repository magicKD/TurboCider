all: build/transformer build/transformer-heads build/transformer-private build/transformer-private-gpu build/transformer-startup

CXX := $(shell xcrun --find clang++)
SDK := $(shell xcrun --sdk macosx --show-sdk-path)
# Optional real-checkpoint diagnostic; requires an existing native engine build.
CC := $(shell xcrun --find clang)
LTX_DEPLOYMENT_TARGET ?= 26.2
build/ltx-stage2-replay: src/ltx_stage2_replay.c ../../native/models/ltx_runtime/ltx_native.h ../../native/models/ltx_runtime/ltx_rng.h ../../build/native/libturbocider.dylib Makefile
	mkdir -p build
	$(CC) -x objective-c -std=c11 -D_DARWIN_C_SOURCE -O2 -Wall -Wextra -Werror -isysroot $(SDK) -mmacosx-version-min=$(LTX_DEPLOYMENT_TARGET) $< -L ../../build/native -lturbocider -Wl,-rpath,@loader_path/../../../build/native -framework Foundation -o $@
build/transformer: src/runner.mm
	mkdir -p build
	$(CXX) -std=c++20 -O3 -DNDEBUG -fobjc-arc -isysroot $(SDK) -mmacosx-version-min=15.0 $< -o $@ -framework Foundation -framework Metal -framework MetalPerformanceShaders -framework MetalPerformanceShadersGraph -framework Accelerate -framework CoreML
build/transformer-heads: src/runner_heads.mm
	mkdir -p build
	$(CXX) -std=c++20 -O3 -DNDEBUG -fobjc-arc -isysroot $(SDK) -mmacosx-version-min=15.0 $< -o $@ -framework Foundation -framework Metal -framework MetalPerformanceShaders -framework MetalPerformanceShadersGraph -framework Accelerate -framework CoreML
build/transformer-private: src/runner_private.mm
	mkdir -p build
	$(CXX) -std=c++20 -O3 -DNDEBUG -fobjc-arc -isysroot $(SDK) -mmacosx-version-min=15.0 $< -o $@ -framework Foundation -framework Metal -framework MetalPerformanceShaders -framework MetalPerformanceShadersGraph -framework Accelerate -framework CoreML -framework IOSurface -ldl
build/transformer-private-gpu: src/runner_private_gpu.mm
	mkdir -p build
	$(CXX) -std=c++20 -O3 -DNDEBUG -fobjc-arc -isysroot $(SDK) -mmacosx-version-min=15.0 $< -o $@ -framework Foundation -framework Metal -framework MetalPerformanceShaders -framework MetalPerformanceShadersGraph -framework Accelerate -framework CoreML -framework IOSurface -ldl

src/runner.mm: src/baseline.mm scripts/adapt_baseline.py
	python3 scripts/adapt_baseline.py
src/runner_heads.mm: src/runner.mm src/head_branch.inc scripts/add_heads.py
	python3 scripts/add_heads.py
src/runner_private.mm: src/runner.mm src/private_ffn.inc scripts/add_private.py
	python3 scripts/add_private.py
src/runner_private_gpu.mm: src/runner_private.mm scripts/add_private_gpu.py
	python3 scripts/add_private_gpu.py
build/transformer-startup: src/runner_startup.mm
	mkdir -p build
	$(CXX) -std=c++20 -O3 -DNDEBUG -fobjc-arc -isysroot $(SDK) -mmacosx-version-min=15.0 $< -o $@ -framework Foundation -framework Metal -framework MetalPerformanceShaders -framework MetalPerformanceShadersGraph -framework Accelerate -framework CoreML -framework IOSurface -ldl
src/runner_startup.mm: src/runner_private_gpu.mm scripts/add_startup.py
	python3 scripts/add_startup.py
