# Native Steel attention support

The embedded support source in `native/backends/metal_graph/kernels/vendor/steel_attention_support.hpp` derives from MLX Steel (Copyright © 2024 Apple Inc.; MIT License), as vendored by vpipe at local revision `40706e01583b8087cfbb07c148974b35aa4fa9c1`. The D128 online-softmax adapter is derived from `gpu-kernels/metal/vendored/mlx/backend/metal/kernels/steel/attn/kernels/steel_attention.h` (Copyright © 2024-25 Apple Inc.) and reviewed against vpipe's `gpu-kernels/metal/attention/attn_steel.metal`.

TurboCider changes: embed only the seven support headers below, flatten their local includes and pragma-once directives, use a compact native parameter ABI, retain dynamic tails and physical strides, and use 64-bit logical causal endpoint arithmetic. The original adapter uses BF16/D128/BQ32/BK16/four SIMD groups. The Qwen3-VL encoder extension adds explicit query-to-KV head grouping (GQA) and a second FP32/D80 specialization for zero-padded D72 vision attention, each with explicit causal/noncausal entry points. Batch, arbitrary mask, sinks and block-sparse paths remain omitted. The register tile/MMA, online max/sum/PV rescaling and softmax accumulation algorithms are retained. This source is self-contained at build and runtime; vpipe is not a runtime dependency.

The local vpipe loader carries a `jump` helper from vpipe's block-sparse path. That helper remains in the embedded support but is unused by this adapter. vpipe contributions are Apache-2.0; the project copy of that license is [Apache-2.0.txt](Apache-2.0.txt). No vpipe executable or model is copied. See upstream <https://github.com/tgo-app-dev/vpipe> and <https://github.com/ml-explore/mlx>.

Original copied support-file SHA-256 values:

```
b03cea6a7d5cfe814e6838d8536e2aacdcafd2744cb408e954fda336ce21759a  mlx/backend/metal/kernels/steel/defines.h
d4c36298145d1c5617a94f07dbfabf6ac932afe2fd7ea0c9291f1fb385c5404a  mlx/backend/metal/kernels/steel/utils.h
4766c5a3809d3cf7740e63acc5703ef76b6b4e85d8baf16602c9d517a2fd426c  mlx/backend/metal/kernels/steel/utils/type_traits.h
5a4c559c092132b54d3959054fd1fb01bd01adfa65f95d3a60a01b161f11f0e6  mlx/backend/metal/kernels/steel/utils/integral_constant.h
5865a55ec7bfbe47e86fb21e18939ecc7e2b53e6df2214461fbe06be8789cb04  mlx/backend/metal/kernels/steel/attn/transforms.h
35973c3833af0047655e07921a7e45897aba0a2e0ea5c5f62d7e48f1afcd605a  mlx/backend/metal/kernels/steel/attn/loader.h
7f502d66cf30eba95c35faaeed65da3385c62a48716b739fec65ea5130e13a85  mlx/backend/metal/kernels/steel/attn/mma.h
```

Original attention header SHA-256: `2e96b60caf9ecf1275e9a88636a161caaa4bd3e9c05875577e7c4ff44c1f910b`.

MLX license (verbatim):

```
MIT License

Copyright © 2023 Apple Inc.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```
