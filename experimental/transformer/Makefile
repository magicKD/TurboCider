all: build/transformer build/transformer-heads build/transformer-private build/transformer-private-gpu build/transformer-startup

CXX := /Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/clang++
SDK := /Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX.sdk
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
