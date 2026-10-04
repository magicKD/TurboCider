# GGML quantization layouts and decoding expressions

`native/core/gguf_decode.cpp` adapts scalar decoding expressions and block layouts from
`ggml/src/ggml-quants.c` and `ggml/src/ggml-common.h` in ggml-org/llama.cpp,
commit `64e9bceb2c3a856efed96feda784a50947049feb`. No inference engine is linked.
`native/backends/private/ane_w8_kernels.hpp` also adapts these pinned
Q4_0/Q4_K/Q8_0/Q6_K layouts and expressions for GPU decode+rotation staging
(2026-10-03). The upstream MIT notice below applies to these adaptations.
`native/backends/private/ane_w8_kernels.hpp` also adapts these pinned
Q4_0/Q4_K/Q8_0/Q6_K layouts and expressions for GPU decode+rotation staging
(2026-10-03). The upstream MIT notice below applies to these adaptations.
`native/backends/private/ane_w8_kernels.hpp` additionally adapts the same pinned
Q4_0/Q4_K/Q8_0/Q6_K layouts and expressions for checked GPU decode+rotation
staging (2026-10-03). The GPU source retains original physical row addressing;
no upstream engine is linked. The MIT notice below applies to these adaptations.
The reference is fixed for the decoder oracle; production builds do not read a sibling checkout.

Upstream license at that commit:

MIT License

Copyright (c) 2023-2026 The ggml authors

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
