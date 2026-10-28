# Metal 固定版本源码阅读索引

<!-- knowledge
id: technique-metal-sources
type: source
-->

[专题导航](README.md) · [全部来源](../sources/README.md)

只提炼原来源记录中的固定 revision、关键源码入口与阅读重点；不保留文件 hash、机器回执或实验数据，也不声明是最新版本。作者结果与本地历史观察分别阅读。

## beam-sources-20261016

- [tinygrad/tinygrad @ 3d057f05df53](https://github.com/tinygrad/tinygrad/tree/3d057f05df532deca7661bfb2400c6564f96a04a)
- [search.py](https://github.com/tinygrad/tinygrad/blob/3d057f05df532deca7661bfb2400c6564f96a04a/tinygrad/codegen/opt/search.py)
- [postrange.py](https://github.com/tinygrad/tinygrad/blob/3d057f05df532deca7661bfb2400c6564f96a04a/tinygrad/codegen/opt/postrange.py)
- [heuristic.py](https://github.com/tinygrad/tinygrad/blob/3d057f05df532deca7661bfb2400c6564f96a04a/tinygrad/codegen/opt/heuristic.py)
- [__init__.py](https://github.com/tinygrad/tinygrad/blob/3d057f05df532deca7661bfb2400c6564f96a04a/tinygrad/codegen/opt/__init__.py)

## candle-pretrained-sources-20261017

- [huggingface/candle @ 12bb5d1f6350](https://github.com/huggingface/candle/tree/12bb5d1f6350f4ea9827d9086f213feec1f0af6a)
- [Cargo.toml](https://github.com/huggingface/candle/blob/12bb5d1f6350f4ea9827d9086f213feec1f0af6a/Cargo.toml)
- [LICENSE-MIT](https://github.com/huggingface/candle/blob/12bb5d1f6350f4ea9827d9086f213feec1f0af6a/LICENSE-MIT)
- [LICENSE-APACHE](https://github.com/huggingface/candle/blob/12bb5d1f6350f4ea9827d9086f213feec1f0af6a/LICENSE-APACHE)
- [gguf_file.rs](https://github.com/huggingface/candle/blob/12bb5d1f6350f4ea9827d9086f213feec1f0af6a/candle-core/src/quantized/gguf_file.rs)

## candle-sources-20261017

- [huggingface/candle @ 12bb5d1f6350](https://github.com/huggingface/candle/tree/12bb5d1f6350f4ea9827d9086f213feec1f0af6a)
- [Cargo.toml](https://github.com/huggingface/candle/blob/12bb5d1f6350f4ea9827d9086f213feec1f0af6a/Cargo.toml)
- [Cargo.toml](https://github.com/huggingface/candle/blob/12bb5d1f6350f4ea9827d9086f213feec1f0af6a/candle-core/Cargo.toml)
- [Cargo.toml](https://github.com/huggingface/candle/blob/12bb5d1f6350f4ea9827d9086f213feec1f0af6a/candle-nn/Cargo.toml)
- [Cargo.toml](https://github.com/huggingface/candle/blob/12bb5d1f6350f4ea9827d9086f213feec1f0af6a/candle-transformers/Cargo.toml)

## cb-profile-sources-20261017

- [CMakeLists.txt](https://github.com/ggml-org/llama.cpp/blob/64e9bceb2c3a856efed96feda784a50947049feb/CMakeLists.txt)
- [CONTRIBUTING.md](https://github.com/ggml-org/llama.cpp/blob/64e9bceb2c3a856efed96feda784a50947049feb/CONTRIBUTING.md)
- [CMakeLists.txt](https://github.com/ggml-org/llama.cpp/blob/64e9bceb2c3a856efed96feda784a50947049feb/ggml/src/ggml-metal/CMakeLists.txt)
- [ggml-metal-context.m](https://github.com/ggml-org/llama.cpp/blob/64e9bceb2c3a856efed96feda784a50947049feb/ggml/src/ggml-metal/ggml-metal-context.m)

## compiler-sources-20261016

- [mlc-ai/mlc-llm @ 8978ea9aec62](https://github.com/mlc-ai/mlc-llm/tree/8978ea9aec626d64096aaa92fbc89bda2e3ea802)
- [mlc-ai/relax @ 201fa0001c81](https://github.com/mlc-ai/relax/tree/201fa0001c81431ccce05419b442c47779ea917c)
- [tinygrad/tinygrad @ 3d057f05df53](https://github.com/tinygrad/tinygrad/tree/3d057f05df532deca7661bfb2400c6564f96a04a)
- [.gitmodules](https://github.com/mlc-ai/mlc-llm/blob/8978ea9aec626d64096aaa92fbc89bda2e3ea802/.gitmodules)
- [pipeline.py](https://github.com/mlc-ai/mlc-llm/blob/8978ea9aec626d64096aaa92fbc89bda2e3ea802/python/mlc_llm/compiler_pass/pipeline.py)
- [fuse_dequantize_matmul_ewise.py](https://github.com/mlc-ai/mlc-llm/blob/8978ea9aec626d64096aaa92fbc89bda2e3ea802/python/mlc_llm/compiler_pass/fuse_dequantize_matmul_ewise.py)
- [low_batch_specialization.py](https://github.com/mlc-ai/mlc-llm/blob/8978ea9aec626d64096aaa92fbc89bda2e3ea802/python/mlc_llm/compiler_pass/low_batch_specialization.py)

## compressed-attention-sources-20261016

- [vllm-project/vllm-metal @ 468b0c8d6a3f](https://github.com/vllm-project/vllm-metal/tree/468b0c8d6a3f2afe617311d76ed25ee5e3b0d05e)
- [turboquant.py](https://github.com/vllm-project/vllm-metal/blob/468b0c8d6a3f2afe617311d76ed25ee5e3b0d05e/vllm_metal/attention/caches/turboquant.py)
- [turboquant_materialize.py](https://github.com/vllm-project/vllm-metal/blob/468b0c8d6a3f2afe617311d76ed25ee5e3b0d05e/vllm_metal/attention/caches/turboquant_materialize.py)
- [turboquant.metal](https://github.com/vllm-project/vllm-metal/blob/468b0c8d6a3f2afe617311d76ed25ee5e3b0d05e/vllm_metal/metal/kernels_v2/turboquant.metal)
- [turboquant.md](https://github.com/vllm-project/vllm-metal/blob/468b0c8d6a3f2afe617311d76ed25ee5e3b0d05e/docs/turboquant.md)

## deepdives-sources-20261016

- [vllm-project/vllm-metal @ 468b0c8d6a3f](https://github.com/vllm-project/vllm-metal/tree/468b0c8d6a3f2afe617311d76ed25ee5e3b0d05e)
- [alibaba/MNN @ dbf9ba61b081](https://github.com/alibaba/MNN/tree/dbf9ba61b08106a3b43426e295d1a29cb41cc8fe)
- [huggingface/candle @ 12bb5d1f6350](https://github.com/huggingface/candle/tree/12bb5d1f6350f4ea9827d9086f213feec1f0af6a)
- [storage.py](https://github.com/vllm-project/vllm-metal/blob/468b0c8d6a3f2afe617311d76ed25ee5e3b0d05e/vllm_metal/attention/caches/storage.py)
- [kv_cache.py](https://github.com/vllm-project/vllm-metal/blob/468b0c8d6a3f2afe617311d76ed25ee5e3b0d05e/vllm_metal/attention/caches/kv_cache.py)
- [block_tables.py](https://github.com/vllm-project/vllm-metal/blob/468b0c8d6a3f2afe617311d76ed25ee5e3b0d05e/vllm_metal/attention/block_tables.py)
- [tensor_bridge.py](https://github.com/vllm-project/vllm-metal/blob/468b0c8d6a3f2afe617311d76ed25ee5e3b0d05e/vllm_metal/tensor_bridge.py)

## llama-pretrained-sources-20261017

- [ggml-org/llama.cpp @ 64e9bceb2c3a](https://github.com/ggml-org/llama.cpp/tree/64e9bceb2c3a856efed96feda784a50947049feb)
- [CONTRIBUTING.md](https://github.com/ggml-org/llama.cpp/blob/64e9bceb2c3a856efed96feda784a50947049feb/CONTRIBUTING.md)
- [CMakeLists.txt](https://github.com/ggml-org/llama.cpp/blob/64e9bceb2c3a856efed96feda784a50947049feb/CMakeLists.txt)
- [CMakeLists.txt](https://github.com/ggml-org/llama.cpp/blob/64e9bceb2c3a856efed96feda784a50947049feb/src/CMakeLists.txt)
- [llama.h](https://github.com/ggml-org/llama.cpp/blob/64e9bceb2c3a856efed96feda784a50947049feb/include/llama.h)

## materializer-sources-20261016

- [vllm-project/vllm-metal @ 468b0c8d6a3f](https://github.com/vllm-project/vllm-metal/tree/468b0c8d6a3f2afe617311d76ed25ee5e3b0d05e)
- [turboquant.py](https://github.com/vllm-project/vllm-metal/blob/468b0c8d6a3f2afe617311d76ed25ee5e3b0d05e/vllm_metal/attention/caches/turboquant.py)
- [turboquant_materialize.py](https://github.com/vllm-project/vllm-metal/blob/468b0c8d6a3f2afe617311d76ed25ee5e3b0d05e/vllm_metal/attention/caches/turboquant_materialize.py)
- [turboquant.metal](https://github.com/vllm-project/vllm-metal/blob/468b0c8d6a3f2afe617311d76ed25ee5e3b0d05e/vllm_metal/metal/kernels_v2/turboquant.metal)
- [turboquant.md](https://github.com/vllm-project/vllm-metal/blob/468b0c8d6a3f2afe617311d76ed25ee5e3b0d05e/docs/turboquant.md)

## metal4-counter-boundary-sources-20261017

- [mtl4timestampgranularity.json](https://developer.apple.com/tutorials/data/documentation/metal/mtl4timestampgranularity.json)
- [relaxed.json](https://developer.apple.com/tutorials/data/documentation/metal/mtl4timestampgranularity/relaxed.json)
- [precise.json](https://developer.apple.com/tutorials/data/documentation/metal/mtl4timestampgranularity/precise.json)
- [writetimestamp(granularity:counterheap:index:).json](https://developer.apple.com/tutorials/data/documentation/metal/mtl4computecommandencoder/writetimestamp(granularity:counterheap:index:).json)

## metal4-counter-timing-sources-20261017

- [mtl4timestampgranularity.json](https://developer.apple.com/tutorials/data/documentation/metal/mtl4timestampgranularity.json)
- [relaxed.json](https://developer.apple.com/tutorials/data/documentation/metal/mtl4timestampgranularity/relaxed.json)
- [precise.json](https://developer.apple.com/tutorials/data/documentation/metal/mtl4timestampgranularity/precise.json)
- [writetimestamp(granularity:counterheap:index:).json](https://developer.apple.com/tutorials/data/documentation/metal/mtl4computecommandencoder/writetimestamp(granularity:counterheap:index:).json)

## mlc-quant-sources-20261017

- [mlc-ai/mlc-llm @ 8978ea9aec62](https://github.com/mlc-ai/mlc-llm/tree/8978ea9aec626d64096aaa92fbc89bda2e3ea802)
- [mlc-ai/relax @ 201fa0001c81](https://github.com/mlc-ai/relax/tree/201fa0001c81431ccce05419b442c47779ea917c)
- [__init__.py](https://github.com/mlc-ai/mlc-llm/blob/8978ea9aec626d64096aaa92fbc89bda2e3ea802/python/mlc_llm/__init__.py)
- [base.py](https://github.com/mlc-ai/mlc-llm/blob/8978ea9aec626d64096aaa92fbc89bda2e3ea802/python/mlc_llm/base.py)
- [attach_cuda_graph_alloc_init_func.py](https://github.com/mlc-ai/mlc-llm/blob/8978ea9aec626d64096aaa92fbc89bda2e3ea802/python/mlc_llm/compiler_pass/attach_cuda_graph_alloc_init_func.py)
- [attach_embedding_allocator.py](https://github.com/mlc-ai/mlc-llm/blob/8978ea9aec626d64096aaa92fbc89bda2e3ea802/python/mlc_llm/compiler_pass/attach_embedding_allocator.py)

## mlc-vm-stages-sources-20261017

- [mlc-ai/relax @ 201fa0001c81](https://github.com/mlc-ai/relax/tree/201fa0001c81431ccce05419b442c47779ea917c)
- [apache/tvm-ffi @ df463f9bd269](https://github.com/apache/tvm-ffi/tree/df463f9bd269867fc7f4d578d62d80cc2a11d004)
- [mlc-ai/mlc-llm @ 8978ea9aec62](https://github.com/mlc-ai/mlc-llm/tree/8978ea9aec626d64096aaa92fbc89bda2e3ea802)
- [vm.py](https://github.com/mlc-ai/relax/blob/201fa0001c81431ccce05419b442c47779ea917c/python/tvm/runtime/vm.py)
- [vm.cc](https://github.com/mlc-ai/relax/blob/201fa0001c81431ccce05419b442c47779ea917c/src/runtime/vm/vm.cc)
- [executable.cc](https://github.com/mlc-ai/relax/blob/201fa0001c81431ccce05419b442c47779ea917c/src/runtime/vm/executable.cc)
- [builtin.cc](https://github.com/mlc-ai/relax/blob/201fa0001c81431ccce05419b442c47779ea917c/src/runtime/vm/builtin.cc)

## mlx-bf16-attribution-sources-20261017

- [matmul.cpp](https://github.com/ml-explore/mlx/blob/64ea011cb65f14d9ce2737e60db9a4ae91ed7441/mlx/backend/metal/matmul.cpp)
- [gemv.h](https://github.com/ml-explore/mlx/blob/64ea011cb65f14d9ce2737e60db9a4ae91ed7441/mlx/backend/metal/kernels/gemv.h)
- [gemv.metal](https://github.com/ml-explore/mlx/blob/64ea011cb65f14d9ce2737e60db9a4ae91ed7441/mlx/backend/metal/kernels/gemv.metal)
- [scaled_dot_product_attention.cpp](https://github.com/ml-explore/mlx/blob/64ea011cb65f14d9ce2737e60db9a4ae91ed7441/mlx/backend/metal/scaled_dot_product_attention.cpp)

## mlx-cb-profile-sources-20261017

- [stream.cpp](https://github.com/ml-explore/mlx/blob/64ea011cb65f14d9ce2737e60db9a4ae91ed7441/python/src/stream.cpp)
- [transforms.cpp](https://github.com/ml-explore/mlx/blob/64ea011cb65f14d9ce2737e60db9a4ae91ed7441/python/src/transforms.cpp)
- [metal.cpp](https://github.com/ml-explore/mlx/blob/64ea011cb65f14d9ce2737e60db9a4ae91ed7441/python/src/metal.cpp)
- [stream.h](https://github.com/ml-explore/mlx/blob/64ea011cb65f14d9ce2737e60db9a4ae91ed7441/mlx/stream.h)

## mlx-dispatch-sources-20261017

- [device.cpp](https://github.com/ml-explore/mlx/blob/64ea011cb65f14d9ce2737e60db9a4ae91ed7441/mlx/backend/metal/device.cpp)
- [device.h](https://github.com/ml-explore/mlx/blob/64ea011cb65f14d9ce2737e60db9a4ae91ed7441/mlx/backend/metal/device.h)
- [utils.h](https://github.com/ml-explore/mlx/blob/64ea011cb65f14d9ce2737e60db9a4ae91ed7441/mlx/backend/metal/utils.h)
- [matmul.cpp](https://github.com/ml-explore/mlx/blob/64ea011cb65f14d9ce2737e60db9a4ae91ed7441/mlx/backend/metal/matmul.cpp)

## mlx-lm-batch-sources-20261017

- [pyproject.toml](https://github.com/ml-explore/mlx-lm/blob/93e6d4a6447b3080861985947f0820e76fcb957d/pyproject.toml)
- [__init__.py](https://github.com/ml-explore/mlx-lm/blob/93e6d4a6447b3080861985947f0820e76fcb957d/mlx_lm/__init__.py)
- [generate.py](https://github.com/ml-explore/mlx-lm/blob/93e6d4a6447b3080861985947f0820e76fcb957d/mlx_lm/generate.py)
- [generate_utils.py](https://github.com/ml-explore/mlx-lm/blob/93e6d4a6447b3080861985947f0820e76fcb957d/mlx_lm/generate_utils.py)

## mlx-prefill-schedule-sources-20261017

- [matmul.cpp](https://github.com/ml-explore/mlx/blob/64ea011cb65f14d9ce2737e60db9a4ae91ed7441/mlx/backend/metal/matmul.cpp)
- [gemv.h](https://github.com/ml-explore/mlx/blob/64ea011cb65f14d9ce2737e60db9a4ae91ed7441/mlx/backend/metal/kernels/gemv.h)
- [gemv.metal](https://github.com/ml-explore/mlx/blob/64ea011cb65f14d9ce2737e60db9a4ae91ed7441/mlx/backend/metal/kernels/gemv.metal)
- [scaled_dot_product_attention.cpp](https://github.com/ml-explore/mlx/blob/64ea011cb65f14d9ce2737e60db9a4ae91ed7441/mlx/backend/metal/scaled_dot_product_attention.cpp)

## mlx-runtime-sources-20261016

- [vllm-project/vllm-metal @ 468b0c8d6a3f](https://github.com/vllm-project/vllm-metal/tree/468b0c8d6a3f2afe617311d76ed25ee5e3b0d05e)
- [tmc/gputrace @ 3d7d32e66605](https://github.com/tmc/gputrace/tree/3d7d32e66605ba994fb0765bf9f287682bd7ccc7)
- [turboquant.py](https://github.com/vllm-project/vllm-metal/blob/468b0c8d6a3f2afe617311d76ed25ee5e3b0d05e/vllm_metal/attention/caches/turboquant.py)
- [turboquant_materialize.py](https://github.com/vllm-project/vllm-metal/blob/468b0c8d6a3f2afe617311d76ed25ee5e3b0d05e/vllm_metal/attention/caches/turboquant_materialize.py)
- [turboquant.metal](https://github.com/vllm-project/vllm-metal/blob/468b0c8d6a3f2afe617311d76ed25ee5e3b0d05e/vllm_metal/metal/kernels_v2/turboquant.metal)
- [turboquant.md](https://github.com/vllm-project/vllm-metal/blob/468b0c8d6a3f2afe617311d76ed25ee5e3b0d05e/docs/turboquant.md)

## mlx-stage-counter-sources-20261017

- [device.cpp](https://github.com/ml-explore/mlx/blob/64ea011cb65f14d9ce2737e60db9a4ae91ed7441/mlx/backend/metal/device.cpp)
- [device.h](https://github.com/ml-explore/mlx/blob/64ea011cb65f14d9ce2737e60db9a4ae91ed7441/mlx/backend/metal/device.h)
- [utils.h](https://github.com/ml-explore/mlx/blob/64ea011cb65f14d9ce2737e60db9a4ae91ed7441/mlx/backend/metal/utils.h)
- [matmul.cpp](https://github.com/ml-explore/mlx/blob/64ea011cb65f14d9ce2737e60db9a4ae91ed7441/mlx/backend/metal/matmul.cpp)

## mnn-cache-sources-20261016

- [alibaba/MNN @ dbf9ba61b081](https://github.com/alibaba/MNN/tree/dbf9ba61b08106a3b43426e295d1a29cb41cc8fe)
- [Interpreter.hpp](https://github.com/alibaba/MNN/blob/dbf9ba61b08106a3b43426e295d1a29cb41cc8fe/include/MNN/Interpreter.hpp)
- [Tensor.hpp](https://github.com/alibaba/MNN/blob/dbf9ba61b08106a3b43426e295d1a29cb41cc8fe/include/MNN/Tensor.hpp)
- [MNNForwardType.h](https://github.com/alibaba/MNN/blob/dbf9ba61b08106a3b43426e295d1a29cb41cc8fe/include/MNN/MNNForwardType.h)
- [Interpreter.cpp](https://github.com/alibaba/MNN/blob/dbf9ba61b08106a3b43426e295d1a29cb41cc8fe/source/core/Interpreter.cpp)

## mnn-engine-sources-20261016

- [alibaba/MNN @ dbf9ba61b081](https://github.com/alibaba/MNN/tree/dbf9ba61b08106a3b43426e295d1a29cb41cc8fe)
- [CMakeLists.txt](https://github.com/alibaba/MNN/blob/dbf9ba61b08106a3b43426e295d1a29cb41cc8fe/CMakeLists.txt)
- [CMakeLists.txt](https://github.com/alibaba/MNN/blob/dbf9ba61b08106a3b43426e295d1a29cb41cc8fe/source/backend/metal/CMakeLists.txt)
- [Interpreter.hpp](https://github.com/alibaba/MNN/blob/dbf9ba61b08106a3b43426e295d1a29cb41cc8fe/include/MNN/Interpreter.hpp)
- [MNNForwardType.h](https://github.com/alibaba/MNN/blob/dbf9ba61b08106a3b43426e295d1a29cb41cc8fe/include/MNN/MNNForwardType.h)

## native-kv-sources-20261016

- [vllm-project/vllm-metal @ 468b0c8d6a3f](https://github.com/vllm-project/vllm-metal/tree/468b0c8d6a3f2afe617311d76ed25ee5e3b0d05e)
- [turboquant.py](https://github.com/vllm-project/vllm-metal/blob/468b0c8d6a3f2afe617311d76ed25ee5e3b0d05e/vllm_metal/attention/caches/turboquant.py)
- [turboquant_materialize.py](https://github.com/vllm-project/vllm-metal/blob/468b0c8d6a3f2afe617311d76ed25ee5e3b0d05e/vllm_metal/attention/caches/turboquant_materialize.py)
- [turboquant.metal](https://github.com/vllm-project/vllm-metal/blob/468b0c8d6a3f2afe617311d76ed25ee5e3b0d05e/vllm_metal/metal/kernels_v2/turboquant.metal)
- [turboquant.md](https://github.com/vllm-project/vllm-metal/blob/468b0c8d6a3f2afe617311d76ed25ee5e3b0d05e/docs/turboquant.md)

## pytorch-mps-sources-20261017

- [pytorch/pytorch @ 5c4886908584](https://github.com/pytorch/pytorch/tree/5c4886908584029761b579af026dcfb627c84070)
- [Attention.mm](https://github.com/pytorch/pytorch/blob/5c4886908584029761b579af026dcfb627c84070/aten/src/ATen/native/mps/operations/Attention.mm)
- [Attention.metal](https://github.com/pytorch/pytorch/blob/5c4886908584029761b579af026dcfb627c84070/aten/src/ATen/native/mps/kernels/Attention.metal)
- [PrefillAttention.h](https://github.com/pytorch/pytorch/blob/5c4886908584029761b579af026dcfb627c84070/aten/src/ATen/native/mps/kernels/PrefillAttention.h)
- [MppAttention.h](https://github.com/pytorch/pytorch/blob/5c4886908584029761b579af026dcfb627c84070/aten/src/ATen/native/mps/kernels/MppAttention.h)

## qwen-quality-sources-20261017

- [README.md](https://github.com/ggml-org/llama.cpp/blob/64e9bceb2c3a856efed96feda784a50947049feb/tools/perplexity/README.md)
- [perplexity.cpp](https://github.com/ggml-org/llama.cpp/blob/64e9bceb2c3a856efed96feda784a50947049feb/tools/perplexity/perplexity.cpp)
- [get-wikitext-2.sh](https://github.com/ggml-org/llama.cpp/blob/64e9bceb2c3a856efed96feda784a50947049feb/scripts/get-wikitext-2.sh)
- [qwen.py](https://github.com/ggml-org/llama.cpp/blob/64e9bceb2c3a856efed96feda784a50947049feb/conversion/qwen.py)

## research-sources

- [ml-explore/mlx @ 36969c094354](https://github.com/ml-explore/mlx/tree/36969c094354086e15d2a327a9c50feb20b9cf11)
- [ggml-org/llama.cpp @ 64e9bceb2c3a](https://github.com/ggml-org/llama.cpp/tree/64e9bceb2c3a856efed96feda784a50947049feb)
- [philipturner/metal-flash-attention @ 8671cddc38f1](https://github.com/philipturner/metal-flash-attention/tree/8671cddc38f19a6eadb804dee6a3ca2954b8bf32)
- [device.cpp](https://github.com/ml-explore/mlx/blob/36969c094354086e15d2a327a9c50feb20b9cf11/mlx/backend/metal/device.cpp)
- [matmul.cpp](https://github.com/ml-explore/mlx/blob/36969c094354086e15d2a327a9c50feb20b9cf11/mlx/backend/metal/matmul.cpp)
- [quantized.cpp](https://github.com/ml-explore/mlx/blob/36969c094354086e15d2a327a9c50feb20b9cf11/mlx/backend/metal/quantized.cpp)
- [quantized_nax.h](https://github.com/ml-explore/mlx/blob/36969c094354086e15d2a327a9c50feb20b9cf11/mlx/backend/metal/kernels/quantized_nax.h)

## tvm-aot-sources-20261016

- [mlc-ai/relax @ 201fa0001c81](https://github.com/mlc-ai/relax/tree/201fa0001c81431ccce05419b442c47779ea917c)
- [apache/tvm-ffi @ df463f9bd269](https://github.com/apache/tvm-ffi/tree/df463f9bd269867fc7f4d578d62d80cc2a11d004)
- [CMakeLists.txt](https://github.com/mlc-ai/relax/blob/201fa0001c81431ccce05419b442c47779ea917c/CMakeLists.txt)
- [.gitmodules](https://github.com/mlc-ai/relax/blob/201fa0001c81431ccce05419b442c47779ea917c/.gitmodules)
- [pyproject.toml](https://github.com/mlc-ai/relax/blob/201fa0001c81431ccce05419b442c47779ea917c/pyproject.toml)
- [__init__.py](https://github.com/mlc-ai/relax/blob/201fa0001c81431ccce05419b442c47779ea917c/python/tvm/__init__.py)
