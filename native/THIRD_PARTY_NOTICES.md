# Third-party notices

The FLUX.2/Qwen3/VAE implementation was ported with reference to mflux
(commit 12fd27ea7015c6c872ced51b56b313306a543dd2). Model weights are not distributed.

MIT License

Copyright (c) 2026 Filip Strand

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


FastMetal's optional worker imports FastVideo and its TAEHV decoder from an
external installation selected by the user profile. TurboCider does not copy
those Python sources or model weights into this repository. A distributable
FastMetal package must include the corresponding FastVideo/TAEHV Apache-2.0
and third-party NOTICE files from the exact installation it bundles; the
repository's `profiles/fastmetal.example.json` is intentionally only a
portable configuration example.

MLX is Copyright Apple Inc., MIT-licensed. Its unmodified native libraries
are build dependencies; include their original license when packaging them.

The optional GGUF image backend packages an unmodified stable-diffusion.cpp
`sd-cli`/`sd-server` build pinned by `tools/native/install_sd_cpp.py`. The
currently pinned Apple Silicon archive is built by the Unsloth mirror from
source commit `13b9d92b5e9a1563536c9c980e700470f9ab6702`. stable-diffusion.cpp is
Copyright (c) 2023 leejet and MIT-licensed. The package includes the original
license as `SD-CPP-LICENSE.txt`; model weights remain external user assets.

H3 native kernels and model components are derived from h3.c-fork.
The source copy is modified for library configuration, resource discovery and
native Apple media I/O. Its license follows:

TurboCider also vendors the LoRA preparation implementations
`tools/native/merge_h3_lora.py` and `tools/native/merge_ltx_refiner.py` from
the corresponding `h3.c/tools/` sources. The vendored files are kept
byte-identical to the audited source revisions below so that packaged
TurboCider installations do not need a sibling `h3.c` checkout:

* `merge_h3_lora.py` SHA-256
  `91bbb83eff93f2d97b8f53f62bf711f34eeef0ee438bb3b1244d5bc33b677703`
* `merge_ltx_refiner.py` SHA-256
  `e473d53681ae89cae83fce5253a18b7239e66bd701c314308a5eb210e68f6051`

These scripts are covered by the same MIT license and copyright notice as the
derived H3 tooling below.

MIT License

Copyright (c) 2026 Salvatore Sanfilippo

The experimental Apple Neural Engine bridge and related ANE paths are also
Copyright (c) 2026 Manjeet Singh. The private bridge sources are retained only
under `experimental/video/h3/vendor`; they are not linked into the shipping
TurboCider library. The shipping H3 runtime uses a fail-closed stub and public
Core ML where applicable.

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


The excluded LTX Gemma prototype and its synthetic parity fixture reference
ltx-2-mlx commit 91e6f6c9bd621ff2ae31adfee643e113d67d6ae8.

MIT License

Copyright (c) 2025 dgrauet

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
