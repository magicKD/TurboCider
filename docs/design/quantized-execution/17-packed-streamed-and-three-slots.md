# 17 · packed-streamed 与三槽前瞻实验（2026-10-01）

[目录](README.md) · [内存合同](03-bounded-dequant-and-memory.md) · [执行合同](10-execution-and-product-integration.md)

本阶段实现 R3 的 **压缩权重流式读取、p=2/三槽** 子包，接入真实 Qwen3 与 Z，继续
复用原 StageExecutor/IoExecutor/reader/ledger。不是全部R3/低容量硬件/whole-request
认证；tile、更小预算和Z refiners等仍须推进，不用这个实验放宽生产catalog资格。

后续 [18](18-streamed-refiners-and-bank-boundaries.md) 已增加浮点 refiners 流式银行与
释放交接；本页的常驻 refiner 数字保留为本阶段真实历史证据，不改写旧回执。

## 1. 实际读取与 ownership

`GgufWeightPager`读取descriptor中的明确`source_residency`，默认packed_resident不变：

- packed_streamed不构造每层compressed MLX数组；只保留必要resident aliases，以及
  **一个1 MiB** allocator-backed read buffer，owner提前分配/捕获raw pointer，计入同ledger。
  worker只pread与同步decode，不创建array、操作MLX stream或扩容buffer。
- 每个source按完整row、固定最大rows/chunk读取；过大的单row明确报floor，不偷偷
  分配整projection。目标写在既有独占slot相应row span；CPU函数返回后buffer可复用。
- 相同source的affine codes/scales/biases在descriptor阶段组成一个SourceTask：同chunk
  读取一次，再直填三份已计划target。不会每个part重新读整tensor，亦无隐藏packed大temporary。
- CPU decoder本身不增加I/O责任，仍使用checked原block/strides/RNE API。浮点compat
  BF16→F16在source原始bytes上decode一次，不把未转换BF16伪装成F16。
- source held fd与native proof不按路径重开；fill前/后、pass barrier及gather revalidate。
  cancel/short/changed source不发布Ready，旧ticket也不可bind。并发stream buffer借用失败，
  不覆盖其他reader；当前streamed版本严格要求Q=1，后续多worker须独立buffer协议。
- gather只读取真实token IDs对应row，重复/顺序保持；不缓存整个词表。输出独立backing
  与claim，最后消费者完成再释放。resident field中packed embedding仍不伪装dense矩阵。

计划保存source mode/固定buffer容量、layer order/slots/dtypes，cache identity含source mode。
metrics分别报告logical source、retained packed capacity、read buffer capacity、实际successful
payload read bytes、I/O active、decode active、exposed wait；不把重叠时间相加当wall。
错误/取消前实际读过的bytes也计数。metadata/首次内容hash的I/O仍不混进payload fill计数。

## 2. p=0/1/2 与实际资源

新实验配置允许`prefetch_layers=2`，含义仍为当前层之外两层，总slots=3，G1/P0/D2/Q1。
strict JSON/enum/type/conflict规则保留；p3、unknown mode、requant/ANE/旧预算混用仍拒绝。
private probe/layout支持不代表public generate已qualified。manual布局不一致不会被自动重写。

Qwen3源由三份lease/config/tokenizer绑定：

```sh
TURBOCIDER_QWEN3_GGUF_SOURCE_RESIDENCY=packed_streamed
TURBOCIDER_QWEN3_GGUF_PREFETCH=2
TURBOCIDER_QWEN3_GGUF_WEIGHT_LIMIT_BYTES=805306368
```

以上仍是明确实验环境设置，其他必要GGUF/config路径见15；805306368是768MiB
**managed weights ceiling**，不是process RAM上界。encoder p0/p1/p2也分别在256/512/768MiB
下通过；相同256MiB下packed_resident正确拒绝，不自动切模式或少slots。

Z schema2 execution.quantized_execution已存在的source字段可显式`packed_streamed`，
仍须experimental build；runner新增`--source-residency packed_streamed --prefetch 2`。
precision profile与之前相同，既有quality/默认/生产门禁不改。

## 3. 真实 encoder 12格证据

Qwen3-4B Q8与mixed Q4_K/Q6_K，两source modes×三种prefetch，共12次21-token conditioning。
每种文件的六格输出SHA完全相同；各35 fills/35compute，slots真实为1/2/3。
streamed没有全压缩模型驻留，read_buffer=1,048,576 bytes：

| 文件 | source mode | p0 managed peak | p1 managed peak | p2 managed peak |
| --- | --- | ---: | ---: | ---: |
| Q8 | resident | 4,369,174,528 | 4,571,046,912 | 4,772,919,296 |
| mixed K | resident | 2,629,668,864 | 2,831,541,248 | 3,033,413,632 |
| Q8 | streamed | 203,035,648 | 404,908,032 | 606,780,416 |
| mixed K | streamed | 同上 | 同上 | 同上 |

每dense槽计划容量201,916,416 bytes，三槽总605,749,248；actual allocator capacity可小于
aligned reservation upper，不能将两者强行等同。gather与read buffer独立计费。
source原始embedding仅取21rows，raw全表不展开也不load；一pass实际按所需block顺序读取。

[encoder回执](validation/gguf-streamed-encoder-20261001.json)包括各SHA、actual I/O、slots/
capacity、raw receipt SHA以及两文件的resident小预算拒绝。不是whole-request hard cap证明。

## 4. Z 与组合请求

真实Z Q8/compat-affine、512²/seed42/4steps、p2 streamed：

- retained packed=1,455,429,888 bytes/capacity1,455,438,080：仍是fixed/refiner原浮点银行。
  main block compressed weights不缓存。**不是所有Z参数都已按一小片streamed**，也不支持
  量化refiners；下一阶段需继续拆refiner/live intervals与tiles。
- read buffer1,048,576 bytes，三槽affine target容量611,057,664；managed峰值
  2,067,437,824 bytes。对照resident原source capacity约7.225GB，但不是process peak比较。
- 四steps120fills，payload实际总read=24,532,416,768 bytes：resident fixed初始化+
  四pass main reads，而不是对codes/scales/biases各读一次。首次verify/hash另算。
- initial/所有step/final latent、decoded/PNG vs resident全部exact。同图像SHA，profile不变。
  wall13.32s只是有dump/observer的诊断，不授加速资格，不声称cold SSD throughput。

再将Qwen3 Q8 encoder也设streamed三槽、768MiB ceiling：第5层取消status2且无PNG，
同engine重试完整生成status0；conditioning、所有latent、decoded/PNG与resident组合exact。
encoder权重阶段结束drain/release后进入DiT，两个source/slot银行不在正常交接同时驻留。

[完整请求与取消回执](validation/gguf-streamed-z-20261001.json)，raw媒体/张量/报告在
`outputs/quantized-execution-streamed/`。首个Z probe是在最终I/O计时细化前的二进制，
各case独立保留实际library SHA，不能拿final hash替换先前case身份；数学未改变，最终组合
重試/encoder矩阵与最后源码运行。没有全请求managed/driver envelope或low-capacity认证。

## 5. 验证与复现

- 真实Metal tiny fixture放大到512rows，使16KiB buffer触发多chunk，resident/streamed
  K1/K2/K3、两pass、GPU结果/stale ticket/gather/逃逸claim与释放通过；actual read精确计数。
- 取消fill→partial不可bind→同buffer重新fill成功、malformed容量拒绝；对测试独占temp
  fixture显式截断，source-generation变化导致旧Ready内容失效。没有修改用户模型。
- 最初容量断言错误地把reservation upper当allocator actual；按真实claim范围修正后通过，
  不将初次错误标为硬件通过，也不放宽实际资源预算。
- JSON/plan5项、固定GGML/decoder24项、选定旧GGUF/ANE/ConvRot等31项、Z72项通过；
  既有ANE/CoreML/MLX12项、独立decoder ASan/UBSan、安全程序与compile/syntax/diff通过。
- 最新普通build在实际generate入口再次返回qe_capability_unqualified。发行库/默认模型/
  catalog不替换。source变更不继承旧certified tuple。

```sh
.venv/bin/python tools/validation/run_gguf_streamed_validation.py \
  --q8 models/Qwen3-4B-GGUF/Qwen3-4B-Q8_0.gguf --q4k models/Qwen3-4B-GGUF/Qwen3-4B-Q4_K_M.gguf \
  --config models/Tongyi-MAI-Z-Image-Turbo/text_encoder/config.json \
  --tokenizer models/Tongyi-MAI-Z-Image-Turbo/tokenizer/tokenizer.json \
  --output outputs/new-streamed-encoder
.venv/bin/python tools/native/run_z_image_gguf_quantized.py \
  --library build/quantized-execution/libturbocider.dylib --model models/z-image-runtime-gguf-q8 \
  --output outputs/new-streamed-z --prefetch 2 --source-residency packed_streamed \
  --precision z-mlx-compat-affine-v1 --steps 4 --size 512 --seed 42 --dump
```

## 6. 不得标记完成的范围

一整layer仍是最小execution unit，单packed row必须可装进固定read buffer；更紧预算的
projection/N-tile与K-slice read plan尚未实现。metadata/control/activations/driver envelope、
formal component config、public qualification、refiners、source-quant/media门及M5/ANE/ConvRot
完整路线仍继续原目标；不因source驻留下降就把全部R3或整个线程标为完成。
本阶段无模型下载/缓存删除，磁盘余量约11GiB。
