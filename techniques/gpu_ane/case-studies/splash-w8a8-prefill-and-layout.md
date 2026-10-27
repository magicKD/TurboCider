# Splash：prefill FFN 的 paired-channel W8A8 协同与真实布局

<!-- knowledge
id: technique-gpu-ane-case-studies-splash-w8a8-prefill-and-layout
type: case-study
-->

> 历史案例摘要：用于学习机制、取舍与失败边界；不随库分发实验数据或验证环境，不代表当前实测。

> 实现案例：按机制学习，不按研究批次采用。文中的次数、环境、命令和结果是固定历史实验；旧采集脚本依赖当时的输出布局，不是当前一键复现入口。可运行的离线学习见[方法实验](../examples/README.md)，原始来源和数值证据保留在文内链接。

[专题导航](../README.md) · [事件/校准/重跑](splash-handoff-calibration-and-replay.md) · [固定来源](../sources.md#splash-ffn)

2026-10-17，读本为 `80827e1c3593f7d10c266394e4d4381321b73ad0`，不是“当前 main/最新发布”声明。现有 checkout 的 origin 是 `liang2kl/splash` fork；独立取得的 `incoai/splash` 官方 commit API 返回同一 commit 和 tree `af79421149362f80fc140a2342c9f0cae61da200`。fork、官方 endpoint 和未检查的 default-branch reachability 分开记录，不因 README 链接就偷换来源。

这个 ref 有实际 dense-prefill ANE 路径，而非只有历史实验文案。它同时保留格式、shape、内存、校准收益与私有接口门槛；MoE、mixer、decode 仍走 GPU。本轮只执行原始 CPU 校准代码，不加载私有框架、不运行 Splash Metal/ANE 或模型。

## 1. 切的是 intermediate channel，不是 token 或任意一半矩阵

对 `y = residual + Wd[SiLU(Wg x) * Wu x]`，GPU 取 leading intermediate channels，ANE 取其余：

```text
相同 x ─→ GPU: gate/up 的前 Fg 行 → SwiGLU → down 的前 Fg 输入 ─┐
       └→ ANE: gate/up 的后 Fa 行 → SwiGLU → down 的后 Fa 输入 ─┴→ sum
                                                    residual 只在 GPU 路加一次
```

[`AneFfn.cpp` 402–431](https://github.com/incoai/splash/blob/80827e1c3593f7d10c266394e4d4381321b73ad0/runtime/ops/AneFfn.cpp#L402)以相同 leadingRows 构造 gate/up，down 用 leadingInputs；不把 gate 的前半和 up 的后半配对。ANE 返回 partial down，没有 residual 输入；GPU 的 residual epilogue 后再加 partial，额外一次 BF16 舍入被该实现接受，不声称与 GPU-only 逐位相同。

[`Projection::leadingInputs`](https://github.com/incoai/splash/blob/80827e1c3593f7d10c266394e4d4381321b73ad0/runtime/ops/Linear.cpp#L156)保留原始 `planeInputs_` 物理行宽，调用专门 leading-input kernel。逻辑 K 变小，原 packed planes 的物理 pitch 不会自动变小。GPU 权重是原资产的 view；ANE 部分另由 GPU 解码/旋转/重定标到 int8，不是在 CPU 把整模型预展开为另一份 FP16 checkpoint。

affine planes 里的 `biases` 是量化组的 affine bias，不代表通用 FFN 显式 bias/LoRA 支持。不要把这个 Qwen 专用路径直接套到带 adapter、bias 或不同非线性位置的模型。

## 2. 本 ref 的隐藏维度门槛非常具体

[`unsupported()`](https://github.com/incoai/splash/blob/80827e1c3593f7d10c266394e4d4381321b73ad0/runtime/ops/AneFfn.cpp#L80)和 ABI 常量组合为：

- H 是 2560 的倍数，又须为 `128*(256/32)=1024` 的倍数，且 `H/1024≤8`。
- 因而在正尺寸域，**本组常量实际只容许 H=5120**。这是门槛的代数推导，不是 ANE 硬件只支持该尺寸。
- F 为 512-channel unit 的倍数，至少两 unit；ANE/GPU 都须留下至少一 unit。Qwen-like F=17408 对应 34 unit。
- affine64 可以作 plane views；GGUF 只接受一个非浮点、未预旋转、非嵌套 view 的量化 segment。不能把“GGUF 支持”推广为所有文件/Prism 输入旋转都可 offload。

ANE down 输入每段最多 2560 channel，例如取 8 unit 得 Fa=4096，分成 2560+1536；GPU 仍取 Fg=13312。源码注释给出的分段收益属于作者的设备观察，本机没有复现该速度，也没有据此建立通用 ANE 最佳尺寸表。

## 3. Hadamard、scale 与非线性不能随意换序

GPU 从原 affine Q4/GGUF planes 取值，对输入与 gate/up 权重施加相同的块正交变换 `R = D H / sqrt(n)`；对 SwiGLU 后的 intermediate 和 down 权重也成对旋转。理想实数下 `dot(Rw,Rx)=dot(w,x)`，再做 int8 才引入新的舍入/量化。前一轮 GGML 的同输入 exact 结果不能继承给这里。

输入旋转块为 128，intermediate 块为 512；sign 在 Hadamard 后施加。旋转用于摊开 outliers，不与 SiLU/逐元素乘法交换；不能先旋转 gate/up 再直接当 SwiGLU 输出的旋转。

[`function()`](https://github.com/incoai/splash/blob/80827e1c3593f7d10c266394e4d4381321b73ad0/runtime/ops/AneFfn.cpp#L299)还有关键 scale 顺序：输入 token scale 先恢复 gate 的真实幅度，再进入 SiLU；up 保留缩小的比例，最终 join 再补一次。把 scale 全部搬到非线性外不等价。MIL 用 tanh 形式表达 SiLU；理想数学等价不代表所有硬件 lowering 误差一样。

int8 值先按 `1/128` dequantize 到 FP16 路径，权重 row scale、输入 token scale 和 intermediate token scale 分层恢复。intermediate peak 有 `2^-9` floor，使 FP16 inverse 有限；输出将最后一个 token scale 单独返回，让 GPU 在 FP32 中恢复，避免把完整缩放乘积先塞进 FP16。

因此这里的 W8A8 有明确的输入/权重/动态量化含义；生成 MIL 也可见 FP16 matmul/tanh/conv。不能仅凭 int8 surface 或名字断言每个 op 都是整数 MAC，更不能推导“INT8 在所有尺寸总更快”。

## 4. channel-major surface 与 bucket prefix

函数 bucket 为 512、640……2048，共 13 个；chunk 取最小可容纳函数。校准得到的 minimumRows 可以是 553 之类的非 bucket 边界，此时跑 640-row 函数，只取前 553 行。

[`Surface`](https://github.com/incoai/splash/blob/80827e1c3593f7d10c266394e4d4381321b73ad0/runtime/ane/Program.mm#L391)按 64-byte row stride 与 16KiB 分配页建立 IOSurface，并拒绝系统返回另一 stride/size。ANE 输入/partial 是 channel-major：width 是最大 token 数，行数是 channel；小函数的逻辑 token width 变小，物理 stride 仍是 2048-row backing 的 stride。把它当 compact `channels×chunk_rows` 会读错。

[`wrapSharedMemory`](https://github.com/incoai/splash/blob/80827e1c3593f7d10c266394e4d4381321b73ad0/runtime/metal/MetalBackend.mm#L1176)实际调用 `newBufferWithBytesNoCopy`，检查 owner/整页，并在 Metal deallocator 中持有 owner。这证明所读 GPU wrapper 不复制这份 backing，不证明 ANE 编译器内部没有任何 staging 或整个请求零成本。

pack/join 会按 32-token tile 处理；只在真实 chunk 范围写回/检查有效行，物理表面为最大 bucket 留空间。静态代码分析不等于本轮已验证所有非整除 prefix、线程竞态或 OOB guards。

## 5. 双权重缓冲不是两份全模型

构造时计算各层 row scales；每个 prefill command 先 stage layer0 的 int8 权重，此后每层 stage 下一层。两组 surfaces 按 layer parity 交替：

```text
stage W0/set0 → pack x0 → signal ready0
                         ANE FFN0/set0 与 GPU complement0 + stage W1/set1 可并行
                      → wait done0 → join0 → 下一层 mixer
stage W2/set0 发生在后续层：前一层对 set0 的使用已通过 wait/join
```

输入与 partial surfaces 仍共用一套，不能任意跨层/多请求同时覆写。这里表达了并行机会；没有本机 trace 就不升级成已证明 physical overlap。

纯分配算例 H=5120/F=17408/L=64/ANE=8 unit，按所读 allocate() 规则得 181,731,328 bytes（173.31 MiB）。它只是该类 MetalBuffer/IOSurface 的计划，不是模型 RSS、峰值总内存，也不含 ANE 服务的中间工作区。默认 memory plan 从 KV 容量中为这些 bytes 腾位置；不能只看 W8 文件更小就认为协同更省总 RAM。

一个 Program 提供 13 functions，设计上复用服务侧工作区，而非独立加载 13 个 Program。服务额外内存、冷编译与 reload 仍需实测，不能把编译资产合并自动当一份 resident memory。

## 6. 数值门槛与未授予的资格

[`verify()`](https://github.com/incoai/splash/blob/80827e1c3593f7d10c266394e4d4381321b73ad0/runtime/ops/AneFfn.cpp#L677)用固定合成 normalized BF16 行，检查前两层、每个函数 endpoint、两组权重 buffer；先给 partial 填 NaN 以发现未写输出。比较量为：

`||split - GPU_full||₂ / ||GPU_full - GPU_part||₂ ≤ 0.10`。

分母是 ANE 应贡献的部分，而不是 residual 占主导的整个输出，避免“完全没加 partial 却 whole-output relative error 很小”。零贡献导致不可识别也不据此通过。这个 10% 是该组件策略，不是整模型质量承诺；实际激活分布、后续层、LoRA、任意 prefix 与模型 logits 仍需另验。

本轮只执行原始校准数学和 CPU witness，未运行上述 verify/Metal kernel/private ANE。README 中 GGUF 对 llama.cpp 的一组对照还明确用了 `--disable-ane` 与 BF16 KV，不能把那张速度表宣传成 ANE 贡献。公开发布数据、固定源码设计、本机观察始终分账。
