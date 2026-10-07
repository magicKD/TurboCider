# Public INT8 runtime-weight IO：从 API 到实际预测的正控制

2026-10-07，Asia/Singapore，M4 Max / macOS26.6.2。接续
[寄存器 staging 和视觉对照](ane-register-packed-visual-review-2026-10-07.md)。
本轮改变后续路线的证据是：Public Core ML **确实能够**用 INT8
IOSurface-backed runtime-weight 输入进行实际预测，而不只是 FP16 IO
上的 QDQ 表示。当前主应用的 RuntimeGraph/模型路由未切换为这个新
接口；双后端完整集成、四格 base/真实 LoRA/GGUF/ConvRot、视觉多样本
及 matched GPU E2E 仍需完成。目标仍active，不把组件资格当整模型完成。

## 先排除一个低价值的 Private 图优化

新 `private_ane_qdq_share_benchmark` 对当前 W8 SwiGLU 生成图做严格
research rewrite：gate/up 的相同 activation K slice/dequantize 只保留
一套；不改 weight QDQ、MatMul 数目、累加次序、scale、SiLU、H512
旋转、hidden A8 或 LoRA 边界。rewrite 要求两个声明的 type/shape/
expression 相同，非相同输入拒绝。只在 tools 中使用，**没有推理开关
或默认行为改变**。

12个CPU structural controls覆盖base/LoRA、Sylvester/Comfy、K tile
128/256/1024。实际driver四格每臂5 warmups、31个串行换序samples；
每个paired sample完整packed output/scale/hidden/padding逐位相同，
有效FP16值均finite。准备bindings和CPU满足ready event在计时外；
计时含actual driver submit/finish，不含W/A staging、GPU head/epilogue。

| rows / width / LoRA ABI | original ms | shared QDQ ms | 局部比值 |
| --- | ---: | ---: | ---: |
| 1056 / 5120 / base | 11.258583 | 11.209125 | 1.00441× |
| 1056 / 5120 / LoRA | 11.806708 | 11.784500 | 1.00188× |
| 4224 / 7168 / base | 57.512500 | 57.528708 | 0.99972× |
| 4224 / 7168 / LoRA | 59.907208 | 59.947667 | 0.99933× |

结论：没有可信的实质收益，不接入生产 emitter/scheduler。结果与
compiler可能已有CSE一致，但没有compiler IR/trace，不能断言唯一原因。
同binary串行控制保留全部样本，没有挑热状态/删掉早期慢sample。
observer460次、wall58.7477s、max gap0.15432s，无observation error。
不是正式load隔离或整模型计时。

回执 `outputs/private-ane-qdq-share-components-31samples-20261007.json`，
SHA256 `e4e3c1030dd8fe4ec7550df9b394d8479e285bb028c6f28f0c88ca75395d6a00`。
binary SHA256 `2a01f26953a6cc85db225399958f56d9c18843e7f622fac8bdd253d17f02c325`。

## Public 压缩输入：正确版本和真实 backing

本机SDK的公开 `MLMultiArray.h` 声明：`MLMultiArrayDataTypeInt8` 从
macOS26提供；pixel-buffer initializer支持OneComponent8作为INT8
backing。实际构建/调用没有 `_ANEClient`、`_ANERequest`、private
framework links或未公开编译格式。参考公开
[MLMultiArrayDataTypeInt8](https://developer.apple.com/documentation/coreml/mlmultiarraydatatype/int8)
和 [coremltools 9.0](https://github.com/apple/coremltools/releases/tag/9.0)。

原 `.venv` 的coremltools8.3保留不变：没有macOS26 target/INT8 feature
enum，`TensorType(dtype=np.int8)`明确拒绝；不会拿FP16 slot冒称压缩IO。
另建忽略的 `build/coreml26-tools/`，只用于offline export/测试，固定
coremltools9.0、NumPy1.26.4。不升级项目环境、应用依赖或model文件。

新 exporter `export_runtime_int8_io_probe.py` 生成macOS26 ML Program：

```text
x: INT8[K,M] ── dequantize(1/128) ──┐
                                  ├─ MatMul ─ y: FP16[N,M]
w: INT8[N,K] ── dequantize(1/128) ──┘
```

FP16 control同geometry/normalized values，不含QDQ。两臂的x/w均为
runtime feature，图中无checkpoint/LoRA矩阵const，不是frozen-W模型。
导出前后核验FeatureDataType，output拒绝覆盖，保留official package/
compiled artifact及所有file hashes。rows≤4224、hidden≤4096、
width≤16384，声明FP16 slots≤512MiB；这是fixture边界，不是process
RAM上限或production memory qualification。

新 native public probe用CoreVideo真实pixel buffer创建INT8/FP16
MLMultiArray，逐项核验loaded model的dtype/shape/feature count。输入
保持scoped lock/unlock，输出只用getBytesWithHandler，不永久锁住
output backing。各arm CPU-only/CPU+NE小图均完成21次真实prediction、
44,352个输出值精确通过，覆盖原signed -128与weight A/B/A换绑；
21次返回的output.pixelBuffer均为指定backing。

这是public压缩IO的实际正控制，**不是**ANE实际residency、原生INT8
MAC、GPU/ANE物理overlap或图像生成精度证明。

## 图像 FFN 大形状正控制与探索性时间

再测两个实际FFN大小的MatMul IO geometry，仍使用合成sparse weight
和覆盖全部signed-I8编码的activation。每臂独立process、同native
binary、同declared shapes/normalized source values，CPU+NE policy；
三次weight variant(+64/-64/+64)，每variant7次prediction、前2次warmup
不计，15个hot samples全保留。所有输出逐标量对独立one-hot dense
source oracle精确匹配，A/B/A和backing21次一致。

| M / K / N | FP16 prediction median ms | INT8 prediction median ms | 探索性比值 | FP16 / INT8 slot bytes |
| --- | ---: | ---: | ---: | ---: |
| 1056 / 4096 / 5120 | 7.653250 | 2.929708 | 2.61229× | 61407232 / 36241408 |
| 4224 / 4096 / 7168 | 33.797000 | 12.228125 | 2.76387× | 153878528 / 107216896 |

分别每臂核验113,541,120 / 635,830,272个输出scalar。slot bytes计实际
CVPixelBuffer row pitch×rows，不含CoreML workspace/process RSS；因输出
仍FP16及padding，不把整体slot压缩率称为2×。两图的公开compute plan
都将MatMul/identity规划到NE；INT8图的两个dequantize也preferred NE。
compute plan是anticipated placement，不能把它当execution trace。

这是一条值得接入共享FFN的路线，但上述比值**不是性能资格**：

- 两臂独立process、固定FP16→INT8顺序，没有paired/reverse E2E。
- 合成sparse W，不是实checkpoint；未证明一般dense矩阵同等收益。
- 仅计prediction API，不含GPU decode/rotate/requantize、scales restore、
  完整SwiGLU、GPU complement、真实LoRA、VAE/PNG或模型加载。
- host observer存在但不是strict competing-load gate，也未做process-tree
  memory qualification、设备trace；未测matched optimized GPU。
- 早期未绑定graph的v1/v2回执保留，不与最终bound-v3混为同一provenance。

最初runner逐个启动命令但首个尚running，后三次被benchmark锁拒绝，
没有执行driver；随后确认原handle终结再串行重跑。锁未移除/放宽。
一次public compute-plan helper的ObjC __block捕获导致compile失败，
改成completion后的strong snapshot后构建、实际预测和测试通过；
没有将失败构建当性能样本。

## 组件回执现在也绑定编译图

`run_gpu_component_screen.py --artifact-manifest` 可选绑定完整compiler
receipt，保留原无此参数的兼容路径。运行前/后检查manifest hash、全部
declared文件hash、完整file集合、real files、无symlink/path escape、
file count≤4096及总量≤256MiB。发生digest/缺失/新增/改变时保留失败
观察回执并退出失败，不能只用相同薄binary掩盖换图。

这仍是before/after artifact绑定，不是loaded-image证明或production
RuntimeGraph的immutable snapshot替代。host tests覆盖同长度篡改、
unlisted file、escape/missing/symlink、post-run变化保留失败结果。

最终public binary SHA256
`b09cfc0855fd27e277c31f5e65e74bf53915dc1b186818a9eb203bbbc44dd2c5`。
四个最终回执均artifacts_unchanged=true，observer分别5/4/14/10次，
max gap约0.1306s，binary和完整graph receipts均绑定。文件：

```text
outputs/public-{fp16,int8}-io-m1056-k4096-n5120-ne-bound-v3-20261007.json
outputs/public-{fp16,int8}-io-m4224-k4096-n7168-ne-bound-v3-20261007.json
```

按表格FP16/INT8顺序SHA256：

```text
01a3a663fbb5e8c0d4b5fe5a9168e6b1bf91fc6dbc918b7cc1ed48d5a031a6d6
d97d7959341a65d453f4637cd3b5ce29d3e5157ff864a11a964003aa3b2823a3
76e2577e0310ae877e08346dab9748e8c540352cce9b4b60a49a64e5821b31d4
e77af9a4bc749db0ebb5eaf646b637ca7388cdab5fa501c2ebce88a9605c258d
```

## 回归、重现和真正剩余的工作

旧8.3环境93项selected host regression：92 pass、1 skip（需要新
toolchain的graph类型检查）。隔离9.0环境5项新suite：4 pass、1 skip
（旧8.3拒绝control，已在旧环境实际通过）。新环境实际public integration
test包含FP16/I8×CPU/CPU+NE四次native process；它不是四个独立Python
test，也不是四格整模型验收。`outputs/public-int8-io-{host-regression,
integration}-20261007.log`保留完整记录。两个skip明确且互为环境control。

Native production source未改，前轮Public/Private库各480 source input
hash仍匹配，Public release guard再次通过。selective commits不夹带
用户原ConvRot草稿，references/design/models/adapters保持只读。

可重现的工具环境和小图测试：

```sh
.venv/bin/python -m venv build/coreml26-tools
build/coreml26-tools/bin/python -m pip install coremltools==9.0 numpy==1.26.4
TURBOCIDER_TEST_PUBLIC_INT8_IO=1 build/coreml26-tools/bin/python \
  -m unittest -v tests.native.test_public_runtime_int8_io
```

下一步不是宣布Public W8A8完成，而是将正控制转成完整共享实现：

1. 抽出无private-client依赖的Metal W/A stager与typed IOSurface owner，
   Public/Private用同一raw-GGUF/affine/ConvRot decode/rotate/RNE配方、
   bounded banks/scale cache和content generations；不加完整W副本。
2. Public增加checkpoint-independent完整normalized W8A8 SwiGLU图，
   支持dynamic scales、base/真实LoRA hidden ABI、GPU FP32 partial
   restore/join，并沿用共同scheduler、预算、失败whole-span GPU重算。
3. 确认编译/执行/转换/桥接成本后再做per-operation Public/Private/GPU
   路由；Public保持默认可分发，Private明确授权。不能仅根据plan或
   microbenchmark选固定share，不同时常驻两个backend的同一W副本。
4. 对Z/Qwen 512/1024 base与真实LoRA，以及GGUF/ConvRot完成多prompt/
   seed视觉、matched GPU cold/hot/reverse、load/memory/artifact审计。
   目标为实际快于GPU，不强制旧1.2×；历史N1仅作补充，不重写旧报告。
