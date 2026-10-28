# GPU+ANE 固定版本源码阅读索引

<!-- knowledge
id: technique-gpu-ane-sources
type: source
-->

[专题导航](README.md) · [全部来源](../sources/README.md)

只提炼原来源记录中的固定 revision、关键源码入口与阅读重点；不保留文件 hash、机器回执或实验数据，也不声明是最新版本。作者结果与本地历史观察分别阅读。

## ane-lm-state

- [johnmai-dev/ANE-LM @ 04b1af12aa30](https://github.com/johnmai-dev/ANE-LM/tree/04b1af12aa30b3f280f511d286398469764b4326)
- [huggingface/transformers @ 8498ed425fa0](https://github.com/huggingface/transformers/tree/8498ed425fa01bbee7d086efdf1a6634c2962aed)
- [CMakeLists.txt](https://raw.githubusercontent.com/johnmai-dev/ANE-LM/04b1af12aa30b3f280f511d286398469764b4326/CMakeLists.txt)
- [README.md](https://raw.githubusercontent.com/johnmai-dev/ANE-LM/04b1af12aa30b3f280f511d286398469764b4326/README.md)
- [generate.cpp](https://raw.githubusercontent.com/johnmai-dev/ANE-LM/04b1af12aa30b3f280f511d286398469764b4326/generate.cpp)
- [generate.h](https://raw.githubusercontent.com/johnmai-dev/ANE-LM/04b1af12aa30b3f280f511d286398469764b4326/generate.h)

## apple-coreml-pipeline

- [apple-aiml-research/ml-stable-diffusion @ ea2805dc1945](https://github.com/apple-aiml-research/ml-stable-diffusion/tree/ea2805dc1945be20561c77e5f6d1d9a5a637cda2)
- [LICENSE.md](https://github.com/apple-aiml-research/ml-stable-diffusion/blob/ea2805dc1945be20561c77e5f6d1d9a5a637cda2/LICENSE.md)
- [Package.swift](https://github.com/apple-aiml-research/ml-stable-diffusion/blob/ea2805dc1945be20561c77e5f6d1d9a5a637cda2/Package.swift)
- [ResourceManaging.swift](https://github.com/apple-aiml-research/ml-stable-diffusion/blob/ea2805dc1945be20561c77e5f6d1d9a5a637cda2/swift/StableDiffusion/pipeline/ResourceManaging.swift)
- [ManagedMLModel.swift](https://github.com/apple-aiml-research/ml-stable-diffusion/blob/ea2805dc1945be20561c77e5f6d1d9a5a637cda2/swift/StableDiffusion/pipeline/ManagedMLModel.swift)

阅读重点：MIT notice; no upstream source distributed in topic；package platforms/dependencies; full package not built or resolved。

## ccv-precision

- [liuliu/ccv @ 2de9f60fb93c](https://github.com/liuliu/ccv/tree/2de9f60fb93cda950a8484a59ad14ba4f2a6bb4b)
- [ccv_nnc_mfa_ane_rowwise_coreml.mm](https://github.com/liuliu/ccv/blob/2de9f60fb93cda950a8484a59ad14ba4f2a6bb4b/lib/nnc/mfa/ccv_nnc_mfa_ane_rowwise_coreml.mm)
- [ANERowwiseTransformKernel.cpp](https://github.com/liuliu/ccv/blob/2de9f60fb93cda950a8484a59ad14ba4f2a6bb4b/lib/nnc/mfa/kernels/ANERowwiseTransformKernel.cpp)

阅读重点：original instructions fully reread; no backend forcing or source changes；K-split policy, both dequant scales, FP16 matmul/add, model scale and allowed compute policy。

## ccv-runtime

- [liuliu/ccv @ 2de9f60fb93c](https://github.com/liuliu/ccv/tree/2de9f60fb93cda950a8484a59ad14ba4f2a6bb4b)
- [Package.swift](https://github.com/liuliu/ccv/blob/2de9f60fb93cda950a8484a59ad14ba4f2a6bb4b/Package.swift)
- [ccv_nnc.h](https://github.com/liuliu/ccv/blob/2de9f60fb93cda950a8484a59ad14ba4f2a6bb4b/lib/nnc/ccv_nnc.h)
- [ccv_nnc_mfa.hpp](https://github.com/liuliu/ccv/blob/2de9f60fb93cda950a8484a59ad14ba4f2a6bb4b/lib/nnc/mfa/ccv_nnc_mfa.hpp)
- [ccv_nnc_mfa_defines.hpp](https://github.com/liuliu/ccv/blob/2de9f60fb93cda950a8484a59ad14ba4f2a6bb4b/lib/nnc/mfa/ccv_nnc_mfa_defines.hpp)

阅读重点：same fixed root fully read, no source/backend force or external worktree writes；unpatched SwiftPM C/ObjectiveC++ targets, frameworks and compiler settings。

## compiler-packets

- [infervisor/plow @ 35dfa22ed319](https://github.com/infervisor/plow/tree/35dfa22ed319ece06255e10e4d76996f7a48977b)
- [trymirai/uzu @ 2c16e96796de](https://github.com/trymirai/uzu/tree/2c16e96796de84abccf5ade4ef295f464d2b2a87)
- [mlc-ai/mlc-llm @ 8978ea9aec62](https://github.com/mlc-ai/mlc-llm/tree/8978ea9aec626d64096aaa92fbc89bda2e3ea802)
- [interp.metal](https://github.com/infervisor/plow/blob/35dfa22ed319ece06255e10e4d76996f7a48977b/runtime/apple/interp.metal)
- [mlp_channel.metal](https://github.com/infervisor/plow/blob/35dfa22ed319ece06255e10e4d76996f7a48977b/runtime/apple/mlp_channel.metal)
- [mod.rs](https://github.com/infervisor/plow/blob/35dfa22ed319ece06255e10e4d76996f7a48977b/crates/plowrt/src/exec/apple/mod.rs)
- [hetero.rs](https://github.com/infervisor/plow/blob/35dfa22ed319ece06255e10e4d76996f7a48977b/crates/plowrt/src/exec/apple/hetero.rs)

阅读重点：public owner, main support scope and separate Apple/ASR entry；packet ABI, persistent wait/execute/publication, visibility fences and fault word。

## consumer-handoff

- [tgo-app-dev/vpipe @ c2989a768418](https://github.com/tgo-app-dev/vpipe/tree/c2989a76841888e4d8d208fad4f476c487710962)
- [ane-ffn.h](https://github.com/tgo-app-dev/vpipe/blob/c2989a76841888e4d8d208fad4f476c487710962/generative-models/shared/ane-ffn.h)
- [ane-ffn.cc](https://github.com/tgo-app-dev/vpipe/blob/c2989a76841888e4d8d208fad4f476c487710962/generative-models/shared/ane-ffn.cc)
- [ane-emitter.cc](https://github.com/tgo-app-dev/vpipe/blob/c2989a76841888e4d8d208fad4f476c487710962/apple-silicon/coreml/ane-emitter.cc)
- [ane-module.cc](https://github.com/tgo-app-dev/vpipe/blob/c2989a76841888e4d8d208fad4f476c487710962/generative-models/shared/ane-module.cc)

阅读重点：projection/adaptor row ABI, options and sticky powers-of-four scale；graph selection; merge LoRA before FP16 scale; projection1 bias; retry and full output restoration。

## coreml-delegates

- [microsoft/onnxruntime @ f33c1744e1e1](https://github.com/microsoft/onnxruntime/tree/f33c1744e1e1064d3618d1d0ce533f0b946aed2d)
- [pytorch/executorch @ 3f15285f3b66](https://github.com/pytorch/executorch/tree/3f15285f3b668b0ecdac6fa985bc207edeb50efc)
- [coreml_provider_factory.h](https://github.com/microsoft/onnxruntime/blob/f33c1744e1e1064d3618d1d0ce533f0b946aed2d/include/onnxruntime/core/providers/coreml/coreml_provider_factory.h)
- [narrow.h](https://github.com/microsoft/onnxruntime/blob/f33c1744e1e1064d3618d1d0ce533f0b946aed2d/include/onnxruntime/core/common/narrow.h)
- [coreml_execution_provider.cc](https://github.com/microsoft/onnxruntime/blob/f33c1744e1e1064d3618d1d0ce533f0b946aed2d/onnxruntime/core/providers/coreml/coreml_execution_provider.cc)
- [coreml_options.cc](https://github.com/microsoft/onnxruntime/blob/f33c1744e1e1064d3618d1d0ce533f0b946aed2d/onnxruntime/core/providers/coreml/coreml_options.cc)

阅读重点：MIT license; no upstream implementation copied into wiki；public option meanings, compute units and explicit cache identity responsibility。

## coreml-split-einsum

- [mlcomputeplan-1w21n](https://developer.apple.com/documentation/coreml/mlcomputeplan-1w21n)
- [mlcomputeplandeviceusage](https://developer.apple.com/documentation/coreml/mlcomputeplandeviceusage)

阅读重点：unchanged original three attention functions actually executed with real PyTorch CPU; V2 floor query chunks and missing fast-path mask；export precision/compute units, XL custom/default VAE asymmetry and LUT palettization, not W8A8 MAC proof。

## drawthings-ccv

- [drawthingsai/draw-things-community @ c31f67eecfe2](https://github.com/drawthingsai/draw-things-community/tree/c31f67eecfe2108e2573bc65ed45ce489a8c9b71)
- [liuliu/s4nnc @ ccb9971247b6](https://github.com/liuliu/s4nnc/tree/ccb9971247b64ada98adf58c0028355b7b85fab0)
- [liuliu/ccv @ 2de9f60fb93c](https://github.com/liuliu/ccv/tree/2de9f60fb93cda950a8484a59ad14ba4f2a6bb4b)

阅读重点：complete root instructions, no app/source modification；exact direct ccv/s4nnc dependency pins。

## executorch-mlx

- [pytorch/executorch @ 3f15285f3b66](https://github.com/pytorch/executorch/tree/3f15285f3b668b0ecdac6fa985bc207edeb50efc)

## fastvlm-handoff

- [apple-aiml-research/ml-fastvlm @ 6f7b13179bbc](https://github.com/apple-aiml-research/ml-fastvlm/tree/6f7b13179bbc81e6aa7d921bc7c6335c43904778)
- [LICENSE_MODEL](https://raw.githubusercontent.com/apple-aiml-research/ml-fastvlm/6f7b13179bbc81e6aa7d921bc7c6335c43904778/LICENSE_MODEL)
- [ACKNOWLEDGEMENTS](https://raw.githubusercontent.com/apple-aiml-research/ml-fastvlm/6f7b13179bbc81e6aa7d921bc7c6335c43904778/ACKNOWLEDGEMENTS)
- [README.md](https://raw.githubusercontent.com/apple-aiml-research/ml-fastvlm/6f7b13179bbc81e6aa7d921bc7c6335c43904778/README.md)
- [pyproject.toml](https://raw.githubusercontent.com/apple-aiml-research/ml-fastvlm/6f7b13179bbc81e6aa7d921bc7c6335c43904778/pyproject.toml)

## ggml-ane-lifecycle

- [arozanov/ggml-ane @ fecc9c9ffc9f](https://github.com/arozanov/ggml-ane/tree/fecc9c9ffc9fc60aa5cb569b8caadc000a07e0b0)
- [ggml-org/llama.cpp @ 64e9bceb2c3a](https://github.com/ggml-org/llama.cpp/tree/64e9bceb2c3a856efed96feda784a50947049feb)
- [ggml-ane.h](https://github.com/arozanov/ggml-ane/blob/fecc9c9ffc9fc60aa5cb569b8caadc000a07e0b0/ggml/include/ggml-ane.h)
- [CMakeLists.txt](https://github.com/arozanov/ggml-ane/blob/fecc9c9ffc9fc60aa5cb569b8caadc000a07e0b0/ggml/src/ggml-ane/CMakeLists.txt)
- [ane_bridge.h](https://github.com/arozanov/ggml-ane/blob/fecc9c9ffc9fc60aa5cb569b8caadc000a07e0b0/ggml/src/ggml-ane/ane_bridge.h)
- [ane_bridge.m](https://github.com/arozanov/ggml-ane/blob/fecc9c9ffc9fc60aa5cb569b8caadc000a07e0b0/ggml/src/ggml-ane/ane_bridge.m)

阅读重点：MIT attribution retained; no original implementation distributed；unpinned external GGML build, private API, prefill dimensions and author-only TFLOPS claims。

## ggml-fewrow

- [ggml-org/llama.cpp @ 988190680d5a](https://github.com/ggml-org/llama.cpp/tree/988190680d5a89fce97de3c20df2c2813731fd61)
- [CONTRIBUTING.md](https://github.com/ggml-org/llama.cpp/blob/988190680d5a89fce97de3c20df2c2813731fd61/CONTRIBUTING.md)
- [CMakeLists.txt](https://github.com/ggml-org/llama.cpp/blob/988190680d5a89fce97de3c20df2c2813731fd61/CMakeLists.txt)
- [CMakeLists.txt](https://github.com/ggml-org/llama.cpp/blob/988190680d5a89fce97de3c20df2c2813731fd61/ggml/CMakeLists.txt)
- [CMakeLists.txt](https://github.com/ggml-org/llama.cpp/blob/988190680d5a89fce97de3c20df2c2813731fd61/ggml/src/CMakeLists.txt)

阅读重点：MIT license retained in private build copies; no upstream source distributed in topic；Apple backend scope; no remote install script or model download executed。

## ggml-fusion-timing

- [ggml-org/llama.cpp @ 988190680d5a](https://github.com/ggml-org/llama.cpp/tree/988190680d5a89fce97de3c20df2c2813731fd61)
- [ggml.h](https://github.com/ggml-org/llama.cpp/blob/988190680d5a89fce97de3c20df2c2813731fd61/ggml/include/ggml.h)
- [ggml-backend.h](https://github.com/ggml-org/llama.cpp/blob/988190680d5a89fce97de3c20df2c2813731fd61/ggml/include/ggml-backend.h)
- [ggml-alloc.h](https://github.com/ggml-org/llama.cpp/blob/988190680d5a89fce97de3c20df2c2813731fd61/ggml/include/ggml-alloc.h)
- [ggml-cpp.h](https://github.com/ggml-org/llama.cpp/blob/988190680d5a89fce97de3c20df2c2813731fd61/ggml/include/ggml-cpp.h)

阅读重点：unchanged MIT license already read; retained in private source copy, no upstream code distributed；public status/half ABI, graph metadata and quant type traits; no original quantizer used for the deterministic manual packed fixture。

## litert-coreml

- [google-ai-edge/LiteRT @ b0983dc2dc27](https://github.com/google-ai-edge/LiteRT/tree/b0983dc2dc27141a64fba7553f40299631e65937)
- [README.md](https://github.com/google-ai-edge/LiteRT/blob/b0983dc2dc27141a64fba7553f40299631e65937/tflite/delegates/coreml/README.md)
- [coreml_delegate.h](https://github.com/google-ai-edge/LiteRT/blob/b0983dc2dc27141a64fba7553f40299631e65937/tflite/delegates/coreml/coreml_delegate.h)
- [coreml_delegate.mm](https://github.com/google-ai-edge/LiteRT/blob/b0983dc2dc27141a64fba7553f40299631e65937/tflite/delegates/coreml/coreml_delegate.mm)
- [coreml_delegate_kernel.h](https://github.com/google-ai-edge/LiteRT/blob/b0983dc2dc27141a64fba7553f40299631e65937/tflite/delegates/coreml/coreml_delegate_kernel.h)

阅读重点：Apache-2.0; upstream implementation not redistributed；V2 Compiled Model API, Apple platform table with ANE coming-soon marker, and strict no-new-legacy-app-code directive。

## litert-metal-runtime

- [google-ai-edge/LiteRT @ b0983dc2dc27](https://github.com/google-ai-edge/LiteRT/tree/b0983dc2dc27141a64fba7553f40299631e65937)
- [google-ai-edge/ml-drift @ 1b18771f44c0](https://github.com/google-ai-edge/ml-drift/tree/1b18771f44c0be3d972847b31a80f6dc3d6964b5)
- [WORKSPACE](https://github.com/google-ai-edge/LiteRT/blob/b0983dc2dc27141a64fba7553f40299631e65937/WORKSPACE)
- [CompiledModel.swift](https://github.com/google-ai-edge/LiteRT/blob/b0983dc2dc27141a64fba7553f40299631e65937/litert/swift/Sources/LiteRT/CompiledModel.swift)
- [litert_compiled_model.cc](https://github.com/google-ai-edge/LiteRT/blob/b0983dc2dc27141a64fba7553f40299631e65937/litert/c/litert_compiled_model.cc)
- [ml_drift_metal_accelerator.cc](https://github.com/google-ai-edge/LiteRT/blob/b0983dc2dc27141a64fba7553f40299631e65937/litert/runtime/accelerators/gpu/ml_drift_metal_accelerator.cc)

阅读重点：exact ML Drift dependency commit/archive digest and GL BUILD-only patch; not arbitrary current ML Drift main；reused same-ref platform/API instructions; V2 ANE remains separate from Metal GPU path。

## metal4-public

- [understanding-the-metal-4-core-api](https://developer.apple.com/documentation/metal/understanding-the-metal-4-core-api)
- [resource-synchronization](https://developer.apple.com/documentation/metal/resource-synchronization)
- [machine-learning-passes](https://developer.apple.com/documentation/metal/machine-learning-passes)
- [running-a-machine-learning-model-on-the-gpu-timeline](https://developer.apple.com/documentation/metal/running-a-machine-learning-model-on-the-gpu-timeline)

## metalpetal-fusion

- [aldealabs/raster @ cdba5d0b44fb](https://github.com/aldealabs/raster/tree/cdba5d0b44fb4ffa851e45e50ee7939cb6c62fce)

阅读重点：MIT；actual original Objective-C target and absent dependencies; Swift overlay not built。

## mldrift-int4-fc

- [google-ai-edge/ml-drift @ 1b18771f44c0](https://github.com/google-ai-edge/ml-drift/tree/1b18771f44c0be3d972847b31a80f6dc3d6964b5)
- [CONTRIBUTING.md](https://github.com/google-ai-edge/ml-drift/blob/1b18771f44c0be3d972847b31a80f6dc3d6964b5/CONTRIBUTING.md)
- [.bazelversion](https://github.com/google-ai-edge/ml-drift/blob/1b18771f44c0be3d972847b31a80f6dc3d6964b5/.bazelversion)
- [.bazelrc](https://github.com/google-ai-edge/ml-drift/blob/1b18771f44c0be3d972847b31a80f6dc3d6964b5/.bazelrc)
- [MODULE.bazel](https://github.com/google-ai-edge/ml-drift/blob/1b18771f44c0be3d972847b31a80f6dc3d6964b5/MODULE.bazel)

阅读重点：same Apache-2.0 license read in earlier rounds; unchanged private source copy；upstream project/build scope, not performance reproduction。

## mldrift-quantization

- [google-ai-edge/ml-drift @ 1b18771f44c0](https://github.com/google-ai-edge/ml-drift/tree/1b18771f44c0be3d972847b31a80f6dc3d6964b5)
- [gpu_model_builder.cc](https://github.com/google-ai-edge/ml-drift/blob/1b18771f44c0be3d972847b31a80f6dc3d6964b5/ml_drift/common/gpu_model_builder.cc)
- [fully_connected.cc](https://github.com/google-ai-edge/ml-drift/blob/1b18771f44c0be3d972847b31a80f6dc3d6964b5/ml_drift/common/kernels/fully_connected.cc)
- [fully_connected_util.cc](https://github.com/google-ai-edge/ml-drift/blob/1b18771f44c0be3d972847b31a80f6dc3d6964b5/ml_drift/common/kernels/fully_connected_util.cc)
- [weights_conversion.cc](https://github.com/google-ai-edge/ml-drift/blob/1b18771f44c0be3d972847b31a80f6dc3d6964b5/ml_drift/common/task/weights_conversion.cc)

阅读重点：same Apache-2.0 already read in round10; retained with private source snapshot；shape-sensitive low-bit FC, source quantization route, grouped/sum gates and fused versus conversion cost policy。

## mlx-compiled-ffn

- [ml-explore/mlx @ 7a1d4f5c12ac](https://github.com/ml-explore/mlx/tree/7a1d4f5c12ac82f4b4d0a6e71538d89ca0605247)

阅读重点：MIT attribution, original source stays ignored；fixed-version compile and warm/cache/evaluation contract; author benchmark not inherited。

## mlx-nax-dispatch

- [exploring-llms-mlx-m5](https://machinelearning.apple.com/research/exploring-llms-mlx-m5)

阅读重点：original qmm launch, legacy split policy and original top-level gate；original two-barrier loading and single weight tile。

## mnn-boundaries

- [alibaba/MNN @ dbf9ba61b081](https://github.com/alibaba/MNN/tree/dbf9ba61b08106a3b43426e295d1a29cb41cc8fe)
- [LICENSE.txt](https://github.com/alibaba/MNN/blob/dbf9ba61b08106a3b43426e295d1a29cb41cc8fe/LICENSE.txt)
- [MNNForwardType.h](https://github.com/alibaba/MNN/blob/dbf9ba61b08106a3b43426e295d1a29cb41cc8fe/include/MNN/MNNForwardType.h)
- [npu.md](https://github.com/alibaba/MNN/blob/dbf9ba61b08106a3b43426e295d1a29cb41cc8fe/docs/inference/npu.md)
- [CMakeLists.txt](https://github.com/alibaba/MNN/blob/dbf9ba61b08106a3b43426e295d1a29cb41cc8fe/source/backend/coreml/CMakeLists.txt)

阅读重点：Apache-2.0; upstream implementation not redistributed in wiki；backend identity, shared NN enum and actual precision enum values。

## ncnn-vulkan-policy

- [Tencent/ncnn @ c15904bb8918](https://github.com/Tencent/ncnn/tree/c15904bb8918187ec100f1ebe708942af83b84be)
- [LICENSE.txt](https://github.com/Tencent/ncnn/blob/c15904bb8918187ec100f1ebe708942af83b84be/LICENSE.txt)
- [CMakeLists.txt](https://github.com/Tencent/ncnn/blob/c15904bb8918187ec100f1ebe708942af83b84be/CMakeLists.txt)
- [CMakeLists.txt](https://github.com/Tencent/ncnn/blob/c15904bb8918187ec100f1ebe708942af83b84be/src/CMakeLists.txt)
- [macos.yml](https://github.com/Tencent/ncnn/blob/c15904bb8918187ec100f1ebe708942af83b84be/.github/workflows/macos.yml)

阅读重点：CPU/Vulkan scope; no ANE claim or benchmark inherited；Tencent BSD-3 and separately listed third-party notices; original excerpts remain ignored。

## plow-packet-runtime

- [infervisor/plow @ 35dfa22ed319](https://github.com/infervisor/plow/tree/35dfa22ed319ece06255e10e4d76996f7a48977b)
- [interp.metal](https://github.com/infervisor/plow/blob/35dfa22ed319ece06255e10e4d76996f7a48977b/runtime/apple/interp.metal)
- [dev_isa.h](https://github.com/infervisor/plow/blob/35dfa22ed319ece06255e10e4d76996f7a48977b/runtime/common/dev_isa.h)
- [dev.rs](https://github.com/infervisor/plow/blob/35dfa22ed319ece06255e10e4d76996f7a48977b/crates/packet/src/dev.rs)
- [mod.rs](https://github.com/infervisor/plow/blob/35dfa22ed319ece06255e10e4d76996f7a48977b/crates/plowrt/src/exec/apple/mod.rs)

阅读重点：wire ABI, selected F32 scaled-add/ReLU, real persistent fences/counters/faults and single diagnostic；shared C ABI and asserts, selected opcodes; AMD/CDNA residency comments are not public Metal guarantees。

## runtime-case-studies

- [tgo-app-dev/vpipe @ c2989a768418](https://github.com/tgo-app-dev/vpipe/tree/c2989a76841888e4d8d208fad4f476c487710962)
- [vllm-project/vllm-metal @ 468b0c8d6a3f](https://github.com/vllm-project/vllm-metal/tree/468b0c8d6a3f2afe617311d76ed25ee5e3b0d05e)
- [Epistates/pmetal @ a225c0e2568f](https://github.com/Epistates/pmetal/tree/a225c0e2568f882573b05caeb42d94fed208fdd3)
- [ane-ffn.cc](https://github.com/tgo-app-dev/vpipe/blob/c2989a76841888e4d8d208fad4f476c487710962/generative-models/shared/ane-ffn.cc)
- [ane-ffn.h](https://github.com/tgo-app-dev/vpipe/blob/c2989a76841888e4d8d208fad4f476c487710962/generative-models/shared/ane-ffn.h)
- [ane-worker.h](https://github.com/tgo-app-dev/vpipe/blob/c2989a76841888e4d8d208fad4f476c487710962/apple-silicon/coreml/ane-worker.h)
- [ane-worker.cc](https://github.com/tgo-app-dev/vpipe/blob/c2989a76841888e4d8d208fad4f476c487710962/apple-silicon/coreml/ane-worker.cc)

阅读重点：plan/probe hysteresis, chunk balancing, numerical restore and finish failure；controller window/reprobe and scale constants。

## sdcpp-residency

- [leejet/stable-diffusion.cpp @ 2f886889e6e8](https://github.com/leejet/stable-diffusion.cpp/tree/2f886889e6e8b78738d6b87f7191f6018557c551)
- [leejet/ggml @ 4bf5f6000653](https://github.com/leejet/ggml/tree/4bf5f6000653b7881d00963cd6ddb665ccd62a8d)
- [mtlcommandqueue](https://developer.apple.com/documentation/metal/mtlcommandqueue)

阅读重点：read-only analysis boundaries and excluded model/vocabulary data；no upstream contribution, commit, PR or dependency change。

## splash-command-lifecycle

- [incoai/splash @ 80827e1c3593](https://github.com/incoai/splash/tree/80827e1c3593f7d10c266394e4d4381321b73ad0)
- [CONTRIBUTING.md](https://github.com/incoai/splash/blob/80827e1c3593f7d10c266394e4d4381321b73ad0/CONTRIBUTING.md)
- [Makefile](https://github.com/incoai/splash/blob/80827e1c3593f7d10c266394e4d4381321b73ad0/Makefile)
- [native.mk](https://github.com/incoai/splash/blob/80827e1c3593f7d10c266394e4d4381321b73ad0/dev/native.mk)
- [AwakeClock.hpp](https://github.com/incoai/splash/blob/80827e1c3593f7d10c266394e4d4381321b73ad0/runtime/AwakeClock.hpp)

阅读重点：make check requires supported Metal Mac; no source changes；MSL4/C++20, minimum macOS26.4 and original framework closure。

## splash-ffn

- [incoai/splash @ 80827e1c3593](https://github.com/incoai/splash/tree/80827e1c3593f7d10c266394e4d4381321b73ad0)
- [CONTRIBUTING.md](https://github.com/incoai/splash/blob/80827e1c3593f7d10c266394e4d4381321b73ad0/CONTRIBUTING.md)
- [THIRD_PARTY_NOTICES](https://github.com/incoai/splash/blob/80827e1c3593f7d10c266394e4d4381321b73ad0/THIRD_PARTY_NOTICES)
- [DEVELOPMENT.md](https://github.com/incoai/splash/blob/80827e1c3593f7d10c266394e4d4381321b73ad0/DEVELOPMENT.md)
- [AwakeClock.hpp](https://github.com/incoai/splash/blob/80827e1c3593f7d10c266394e4d4381321b73ad0/runtime/AwakeClock.hpp)

阅读重点：fixed README capability/disable-ane/benchmark scope; not an engine benchmark reproduction；upstream contribution/build guidance; no source changes or external contributions。

## survey

- [ml-explore/mlx @ 36969c094354](https://github.com/ml-explore/mlx/tree/36969c094354086e15d2a327a9c50feb20b9cf11)
- [ggml-org/llama.cpp @ 64e9bceb2c3a](https://github.com/ggml-org/llama.cpp/tree/64e9bceb2c3a856efed96feda784a50947049feb)
- [liuliu/ccv @ 66dc16a3a3b6](https://github.com/liuliu/ccv/tree/66dc16a3a3b688adfa6dc1d19eeeb3071beeb701)
- [device.cpp](https://github.com/ml-explore/mlx/blob/36969c094354086e15d2a327a9c50feb20b9cf11/mlx/backend/metal/device.cpp)
- [quantized.cpp](https://github.com/ml-explore/mlx/blob/36969c094354086e15d2a327a9c50feb20b9cf11/mlx/backend/metal/quantized.cpp)
- [scaled_dot_product_attention.cpp](https://github.com/ml-explore/mlx/blob/36969c094354086e15d2a327a9c50feb20b9cf11/mlx/backend/metal/scaled_dot_product_attention.cpp)
- [ggml-metal-ops.cpp](https://github.com/ggml-org/llama.cpp/blob/64e9bceb2c3a856efed96feda784a50947049feb/ggml/src/ggml-metal/ggml-metal-ops.cpp)

## tinygrad-metal-replay

- [tinygrad/tinygrad @ 3d057f05df53](https://github.com/tinygrad/tinygrad/tree/3d057f05df532deca7661bfb2400c6564f96a04a)

阅读重点：reference remains read-only; original profiling guide read；profiling versus visualization controls; no interactive server launch planned。

## tinygrad-paired-ffn

- [tinygrad/tinygrad @ 3d057f05df53](https://github.com/tinygrad/tinygrad/tree/3d057f05df532deca7661bfb2400c6564f96a04a)

阅读重点：read-only references, no source or driver/system changes；fragment layout and actual MetalF32/FP16/BF16 dtype pairs; notM5 NAX。
