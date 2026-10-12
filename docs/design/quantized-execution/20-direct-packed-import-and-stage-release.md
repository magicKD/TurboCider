# 20 · CPU-direct packed bank、cache 分账与 VAE 交接（2026-10-01）

[目录](README.md) · [预算筛选](19-speed-and-memory-budget-screen.md) · [生命周期合同](10-execution-and-product-integration.md)

本阶段新增**有界读取、事务性直填计算可用 packed bank**，并完成 Z 的 refiner eval /
denoiser→VAE 权重释放。目标是在较大预算下比流式 refill 更快，不依赖全模型 dense
展开。仍是 explicit experimental GPU-only 路线；不是 whole-request hard cap、全部
GGUF 类型/组件资格、INT8 arithmetic 证明或正式 accelerated 认证。

## 1. 先分离 cache 与导入成本

19 的原 native Q8 footprint21.06GB不能全部归因于 raw/repack 双份源。使用同一原
loader 的独立 `mx.set_cache_limit(0)` 对照，峰值变成11.33GB、末尾约8.25GB，PNG
exact，说明 allocator cache 能解释相当一部分额外驻留。此控制在旧二进制执行，
保留其实际 library SHA，不与最终性能样本混池，也不声称剩余每个 byte 已归因。

首版 CPU-direct+cache0峰值约11.16GB；单独添加既有逐 main-block eval 并未降低
整个请求的最高峰。timestamp/sampler 显示最高峰在 denoise 结束后的阶段，prepared
bank 仍跨 VAE 驻留。最终加入 refiner 串行物化与 VAE 前可验证释放后，独立峰值降到
9.562GB；旧版本/失败/负结果全部保留，不改写较早回执。

## 2. 实现与所有权

`GgufPackedBank` 的 constructor 只读目录/冻结 recipes，不分配模型 payload：

- 要求 native generation-bound content proof，使用同 lease 的 held duplicate fd；
  不按路径重开。目录、shape、产出 key 冲突、row floor、类型和 budget 都在执行前检查。
- 支持 Q4_0/Q4_1/Q8_0→MLX affine codes/scales/biases；原codes与FP16scales保留。
  浮点 F32/F16 与 BF16→F16 使用现有 checked CPU decoder，后者明确匹配
  `z-mlx-compat-affine-v1`。不重做量化，不把 BF16 bytes 伪装成 FP16。
- 一个固定1MiB buffer，按完整 row 分块；同一 source chunk 只读一次，再填三份
  affine parts。不会先创建整 tensor CPU staging、全文件 MLX raw bank 或整模型 dense。
- 目标容量逐 field 对齐、checked arithmetic、reserve→actual commit。共享的
  `gguf_storage` 由 array Data 同时持有真实 allocation 和 ledger claim，producer/
  dictionary 释放后，逃逸 view 或 lazy GPU reader 仍持有 claim；原 pager 正例回归通过。
- 所有 tensors 和 source generation 检查成功后才一次性 publish `Weights`。
  early/final cancel、callback failure、short/changed source、nonfinite/overflow 不发布
  partial bank。失败 bank 不能重用；新 bank 可在干净清理后重试。

纯导入器暂不接受 K/IQ、量化 rank1 或不合 binding 的 tensor；不会偷偷 dense fallback。
已有 K decoder/Qwen3 streamed consumer 不因此失去支持，但不自动成为 prepared-bank
consumer。Z 另外通过完整 required namespace/shape 集合验证，没有推测 architecture。

Q8 453 tensors→813 compute fields 的实际资源：

| 字段 | bytes |
| --- | ---: |
| payload 实际读取（不含目录/首次 hash） | 7,224,676,608 |
| 最终 packed + 原浮点 output 逻辑 bytes | 7,563,825,408 |
| planned packed capacity upper | 7,566,213,120 |
| actual packed capacity | 7,564,939,520 |
| read buffer capacity | 1,048,576 |
| managed import peak | 7,565,989,120 |

这里全模型常驻的是**压缩表示及源本来就是浮点的字段**，没有整模型BF16/FP16展开。
source file 仅由 fd/metadata/proof 持有，raw payload 不再作为另一个完整 resident bank。
首次内容验证 bytes/time 与 payload fill 分开，不能把 verification 从 cold total 擦掉。
上述 ledger 不含 encoder/VAE/activations/control/framework/OS，仍不是 RAM cap。

## 3. Z consumer 与严格实验门禁

`TURBOCIDER_Z_GGUF_IMPORT=cpu_direct` 仅在 experimental build 生效，捕获到 model
instance。ordinary build generate 返回 `qe_capability_unqualified`；未知值和混用拒绝。
仅接受 private resident GPU，拒绝 LoRA、ANE、whole-request guard、旧 streamed /
quantized_execution 混用。默认 MLX loader 不改、不注册新发行 catalog tuple。

`TURBOCIDER_Z_GGUF_PACKED_WEIGHT_LIMIT_BYTES` 是独立实验的 **managed weights ceiling**，
默认10GiB；不是新增 whole-request guard。budget 不足不降低 shape/steps或换 source。

最终 consumer revision=`z-serial-refiners-release-before-vae-v1`：

1. 每 pass 按 noise0→context0→noise1→context1 使用原数学路径，每个 refiner `mx::eval`
   完成后再进行下一个，限制同时活跃的转换/激活图。没有改变 refinement 顺序。
2. 最后 denoise `mx::eval` 后 synchronize、revalidate，清空 transformer / cache，
   **检查 ledger storage_bytes=0**，才能记录 `released_before_vae=true` 并进入 VAE。
   无法证明 reader 已结束或 backing 已归还时失败，不扩预算或双份 fallback。
3. source metadata lease 保留到 export 前再次检查；独立 conditioning/latent 不丢弃。
   request cache limit0是 scope，并恢复 caller 原 limit，不修改其他默认请求。

`gguf_import` 单列 source/plan digest、fields/capacities、payload/hash、read/decode/load、
consumer revision 与 release事实。结果 plan 明确 `experimental-unqualified/executable=false`。
这是 compressed W4/W8 + floating MLX compute，不是 native W8A8/ANE 的交付声明。

## 4. 真实 Z 验收与预算速度

M4 Max / 64GiB / macOS26.6.2 / MLX0.32.0。Qwen3 Q8 encoder streamed p2，Z Q8
512² portrait/seed42/4steps；Q4 另测seed20260930、同encoder和同profile。

- Q8 首版/最终 bank、refiner eval、VAE 前释放、main-block eval 诊断的 conditioning、
  initial/every-step/final latent、decoded和PNG均 exact。
- Q4 CPU-direct及最终release版本，与同source原native loader的完整张量/PNG exact。
- tensor180取消 status2且无PNG，同engine重试 status0、完整输出exact。
  最终版本再次复核取消/重试；prepared Q8 在1GiB managed ceiling明确 budget-floor拒绝。
- standalone Metal 对照native loader的所有Q4_0/Q4_1/Q8_0与浮点fields/projections exact；
  lazy GPU reader在producer/dict销毁后仍正确持有claim，最后consumer结束归还到0。
  ASan/UBSan含early/final cancel、retry、一byte不足、unverified、alias冲突、K/type/row
  floor、nonfinite和测试独占file变更。没有修改用户模型。
- 参考准备后fresh toy fixture的ctime在本机变化，首个测试因此正确拒绝旧lease。
  改为参考准备后重新capture并核对SHA未变，未放宽任何source generation规则。

最终同 binary 的两-arm独立 screen（每arm4个warm计时+2个保留warmup，另外memory
新进程；cache miss / fresh engine；未清OS cache）：

| 候选 | 全请求 median 秒 | 观测 footprint 最大值 bytes |
| --- | ---: | ---: |
| streamed+p1 | 10.633 | 3,834,529,016 |
| CPU-direct packed bank + verified VAE release | 9.061 | 9,562,298,752 |

本 cell 的median比值0.8522，只是筛选；样本/会话/ABBA/cold/形状/媒体不满足11的
正式推广门，不称 certified accelerated。预留10%后的经验选择：

| 预算单位 | 6 | 8 | 10 | 16 |
| --- | --- | --- | --- | --- |
| GB（10^9 bytes） | streamed+p1 | streamed+p1 | streamed+p1 | CPU-direct |
| GiB（2^30 bytes） | streamed+p1 | streamed+p1 | CPU-direct | CPU-direct |

10GiB 的观测余量较紧；这些不是真实小容量机器或连续时间 upper/envelope 认证。
原始峰值不减去旧lifetime peak，不用managed权重数字代替process footprint。

## 5. 工具、回归与剩余范围

runner新增 `--native-import cpu_direct --prefetch -1`、显式 managed ceiling、import
取消selector和 `--gpu-eval-blocks`。probe schema仍严格冲突；budget screen绑定
import plan/consumer revision，不混不同binary/consumer或把旧字段倒填为最终身份。

```sh
.venv/bin/python tools/native/run_z_image_gguf_quantized.py \
  --library build/quantized-execution/libturbocider.dylib --model models/z-image-runtime-gguf-q8 \
  --output outputs/new-direct-z --prefetch -1 --native-import cpu_direct --dump \
  --encoder-gguf models/Qwen3-4B-GGUF/Qwen3-4B-Q8_0.gguf \
  --encoder-config models/Tongyi-MAI-Z-Image-Turbo/text_encoder/config.json \
  --encoder-tokenizer models/Tongyi-MAI-Z-Image-Turbo/tokenizer/tokenizer.json
TURBOCIDER_TEST_GPU=1 TURBOCIDER_NATIVE_LIBRARY_DIR=build/quantized-execution \
  TC_STREAMING_SANITIZER=address,undefined .venv/bin/python tests/native/test_gguf_packed_bank.py
```

两个full library builds、streaming六程序ASan/UBSan/3535layouts、固定GGML/decoder24
tests、request5 tests通过；所选probe/budget/旧weights/GGUF回归46通过、10个未启用
opt-in fixtures跳过不计通过。raw在`outputs/quantized-execution-import/`，portable回执
随本阶段保存；无模型下载/用户缓存删除，磁盘仍约11GiB。

[导入/数值/拒绝回执](validation/gguf-direct-packed-diagnostics-20261001.json) ·
[GB预算screen](validation/gguf-direct-budget-gb-20261001.json) ·
[GiB预算screen](validation/gguf-direct-budget-gib-20261001.json)。cache scope另以
134,217,728-byte caller sentinel真实执行验证，request结束后原值恢复；raw为
`final-cache-scope/report.json`，不混进正式计时样本。

更紧预算的tiles、formal component config/whole-request admission与required-site
closure、IQ/其他组件、ConvRot/ANE consumer、static/runtime W8A8、M5条件路线及正式
质量/性能campaign仍须完成。本阶段不以单cell/packed-bank能力替代完整目标。
