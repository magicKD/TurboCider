# 18 · Z refiners 流式化与银行交接（2026-10-01）

[目录](README.md) · [执行合同](10-execution-and-product-integration.md) · [上一阶段](17-packed-streamed-and-three-slots.md)

本阶段继续 R3：消除 packed-streamed Z 仍常驻的约 1.46 GB refiner 权重银行，
同时加速浮点源转换。仍是 private experimental 能力，不是完整 R3、质量资格或
whole-request RAM hard cap；packed-resident 与生产能力门禁不变。

## 1. 真实执行顺序和资源

- packed-streamed descriptor 新增 `refiners` stage，固定 K1/G1/P0/D0/Q1，
  每 pass 严格 `noise0 → context0 → noise1 → context1`。noise/context 具有独立
  layout class，在串行 barrier 释放上一池后才创建下一池。
- `denoiser` 仍为原 30 个 main blocks，p=0/1/2 对应 K1/2/3。
  refiner 与 main backing 银行不同时驻留；固定小字段继续独立 resident。
  两个 pager 各有一个 1 MiB read buffer，全部记入同一 ledger。
- `StageExecutor::release_drained_backing()` 仅允许 owner 在完全 drained 的
  serial/reload 边界释放银行。逻辑 pass/order 不重置，下次 pass 重建 backing。
  carry、retain-all、actual receipt 拒绝；使用过银行交接后也不能补开认证 receipt。
  这不是给旧 certified tuple 增加未认证的释放/重建行为。
- 每个 refiner 继续调用原 `z_block`/`z_context_block`，`mx::eval` 后才退休最后
  reader；caption/image 输出独立持有。取消/错误先 join/drain，再清理/重试。
- descriptor/backend identity 使用 `:refiner-bank-v2` 与
  `interleaved-single-slot-v2`，不能继承旧布局资格。refiner fills/slots/capacity/
  decoded bytes 单列；总 I/O/read/decode active 含两个 pager，不把重叠计时相加当 wall。

streamed recipes 接受量化 refiner 字段，但本次真实 Q8/Q4 fixture 的 refiners 都是
浮点源，**没有真实量化-refiner 模型资格**。量化 stage-fixed 字段仍拒绝。

## 2. 浮点转换优化和原始负结果

第一版逐元素 BF16→FP16 在 refiner 每 pass 重新读取时成为瓶颈：真实 Q8/p2 诊断
wall 20.80 s，decode active 15.48 s；正确输出保留在 `first-q8-p2`，不删除负结果。

`floating_dense_row()` 新增 ARM SIMD F32/F16/BF16 直填，只在连续且对齐的目标
启用，源 byte loads 允许非对齐；其他 layout 继续 scalar。保持 RNE、拒绝非有限源
和目标溢出，不额外 FP32 整层展开、不偷偷 clamp。BF16 SIMD store 也补齐溢出检查。
安全程序通过实际向量 API 对照全部 63,488 个有限 FP16 bit patterns，覆盖
subnormal/±0、FP32 极值、目标 overflow、guard、allocation-free 和 scalar parity。

## 3. 真实 Z 证据

512²、4 steps、compat-affine，Q8 seed42，Q4 seed20260930：

| fixture/layout | managed weights 峰值 bytes | retained fixed bytes | refiner pool bytes |
| --- | ---: | ---: | ---: |
| Q8 streamed p0 | 387,735,808 | 23,750,912 | 361,922,560 |
| Q8 streamed p2 | 636,799,232 | 23,750,912 | 361,922,560 |
| Q4 streamed p2 | 387,633,280 | 23,704,704 | 361,889,792 |

这些是唯一 managed source/slot backing 的计费峰值，不含 encoder、VAE、activations、
MLX cache、driver、metadata/OS。Q4/p2 的峰值由 refiner 而非三份 main packed slots
决定；不能用它宣称全过程只有 0.39 GB。

- 各请求 main fills120、refiner fills16。Q8 main decoded24,433,582,080 bytes；
  refiner decoded5,726,715,904 bytes。Q4 refiner decoded5,725,970,432 bytes。
- Q8 p0/p2 的 conditioning、initial/every-step/final latent、decoded、PNG exact，
  也与上一阶段常驻 refiner 银行 exact。Q4 与前一同源同 profile golden exact。
- Q8 context0/unit1 取消 status2 且无 PNG；同 engine 重试 status0，所有张量/PNG
  与 clean run exact。实际二进制 hash 随各 case 保存，不用最终 hash 替换旧记录。
- Q8 p0/p2 诊断 wall16.77/9.14 s，Q4/p2 11.77 s；有 dumps/observer，cold/cache
  状态不同，**不是配对性能资格**。没有把这两次 wall 的比例作为加速结论。

raw reports 与张量在 `outputs/quantized-execution-refiners/`；portable digest-bound
回执见 [refiner screen](validation/gguf-refiners-20261001.json)。首组 Q8 运行早于
refiner decoded bytes 指标新增，该缺失指标不从后续二进制倒填。

## 4. 验证与复现

新增 executor negatives：重复释放、非 owner、carry、retain-all、receipt 已开启/
释放后再开、银行重建的部分分配失败、失败后不能继续。正常三 pass 重建仍有39 fills，
真实独立 readers 读前/读后 backing 不被提前覆盖。

```sh
TC_STREAMING_SANITIZER=address,undefined .venv/bin/python tests/native/test_streaming_layout.py
TC_GGML_ORACLE_ROOT=../references/llama.cpp TURBOCIDER_TEST_GGUF_MODELS=1 \
  .venv/bin/python tests/native/test_gguf_execution.py
.venv/bin/python tools/native/run_z_image_gguf_quantized.py \
  --library build/quantized-execution/libturbocider.dylib --model models/z-image-runtime-gguf-q8 \
  --output outputs/new-refiner-z --prefetch 0 2 --source-residency packed_streamed \
  --precision z-mlx-compat-affine-v1 --steps 4 --size 512 --seed 42 --dump
```

CPU streaming 六个程序 ASan/UBSan、3535 layouts、固定 GGML/decoder24 tests、
独立 decoder ASan/UBSan 均通过。已有 host/math/Z/ANE 回归记录保留。
最新 experimental/ordinary library 均 build 成功；最新 Q8 streamed 与 resident
完整输出再次 exact，ordinary generate 返回 `qe_capability_unqualified` 且无 PNG。
JSON/plan5 tests、所选旧 GGUF/weights/benchmark31 tests 通过，10 项 opt-in fixture
未启用而 skip，不计通过。误用 pytest 收集 CLI-only helper 的两次调用无测试执行，
没有将其 internal collection error 算作回归通过；M5 专用探针未在 M4 上授予资格。
本阶段不下载模型、不清理用户缓存、不替换 `build/native`。

## 5. 仍须完成与新速度优先级

投影/N tiles、whole-request guard/envelope、正式 per-component request、source-quant/
媒体审核、ConvRot 模型 consumer、静态/runtime W8A8 与 M5 条件路径仍未完成。
普通 BF16 encoder 加载峰值也不能靠较小 DiT 权重池掩盖。

用户新增要求是 6/8/10/16 GB 等预算内尽快运行：后续要比较同 source/profile 的
packed-resident、streamed、p0/p1/p2、合规少量 hot banks，而不把最低内存路线当
所有预算的唯一选择。选择发生在请求前，布局不在执行途中变动；整个模型 dense
展开不因此成为默认路径。经验测得 footprint 不等同完整 envelope 的硬上界。
