# Raw GGUF → 共享 ANE FFN：有界来源窗口与负性能结果

2026-10-07，Asia/Singapore，M4 Max 64GB / macOS26.6.2。接续
[Public ConvRot/GGUF 六格证据](public-w8-convrot-gguf-2026-10-07.md)。
本轮完成新的来源接入和三格真实生成，不是全部目标完成。双后端、
Z/Qwen base/真实 LoRA、512/1024、GGUF/ConvRot、视觉与快于 matched
optimized GPU 的完整要求保持不变；这里的 raw 路线尚未证明盈利。

## 实现与边界

`FfnWeight` 新增显式 `RawGguf{type,columns}`，values 是 rank2 uint8
物理 GGML bytes，不是 uint32 affine codes 或 logical dense matrix。
CPU/FP16 executor 用 `GgufView`，device executor 用对应的
`DeviceWeightEncoding`；严格拒绝混合 affine metadata/ConvRot transform、
错 pitch/columns/type/extent。W8 支持 Q4_0/Q4_K/Q8_0/Q6_K 的既有
共享 Metal decode → register rotation → requantize，不创建 dense W。

Z 的 CPU-direct bank 增加独立 raw source window：最多6矩阵、256 MiB，
owner-thread 同步 pread、verified SourceLease、每次读取/命中均检查
source generation，完整读取成功后才发布。LRU eviction 只移除窗口
引用，escaped/staging readers 的 array Data 继续持有真实 ledger claim；
预算包含这些读者。取消、floor、读失败和来源变化不发布半成品。
raw budget、live/peak、entries、hits/misses/evictions、读字节/时间进入
实际 `gguf_import` receipt。窗口在 VAE 前 drain/clear。

这是 **GPU affine masters + 有界 raw ANE sources**，不是完整模型
raw-resident GPU/ANE consumer；没有新增整模型 raw/dense 副本或改写
checkpoint。bank 的 GPU import registry 仍仅 native Q4_0/Q4_1/Q8_0/
浮点，K-quants 的整模型导入仍未打通；Q4_1 raw W8 不支持。不能把
底层 K decoder 微测覆盖说成整模型 K 支持。

模型/调度层不依赖 private client。原两套 W8 banks/shared events/
prefetch executor 共用这些来源，但 raw 的 CPU read 不是异步预解码，
也不证明物理 GPU/ANE overlap。失败经 `fail_staging` 保留 sticky 原因/
计数，完整 block 在原 GPU 路线上重算；测量/untimed plan 均正常结束。

Z GGUF runtime 的 LoRA 限制从 base-only 扩展为 inference-time：原 base
codes 保持不变，GPU 提供 gate/up correction 和 down-LoRA，graph hidden
ABI 不变。merged LoRA 与 raw checkpoint 来源禁止混用，GPU-only 的
FP16 compute/compiled packed/source validation recipes 仍不与此路线混用。

显式实验开关：

```sh
TURBOCIDER_Z_GGUF_IMPORT=cpu_direct
TURBOCIDER_Z_GGUF_ANE_SOURCE=raw
```

需要 experimental build、resident、runtime GPU+ANE approximation 授权。
默认仍 affine，不根据“支持 raw”自动替换性能更好的来源。Public 仍默认
可分发，Private 仍 opt-in；没有原生 INT8 MAC/FP32 ANE arithmetic 声明。

## 实际 Public Q4 生成

fox/seed42、8步，每格 GPU/runtime 同库独立进程串行。GPU 为原 native
affine consumer，无 raw read；候选为 Public W8、rows、GPU IOSurface、
chunks1/fixed-async1、specialize/scale-cache/launch-fence1、prefetch0。
512/1024 分别352/1056-row 模板，base/LoRA 使用各自 ABI。

| cell | GPU / Public denoise s | GPU / Public request-wall s | calls / headroom / retry / fallback |
| --- | --- | --- | --- |
| Q4 512 base | 9.272510 / 10.651552 | 13.350574 / 15.187231 | 256 / 1 / 0 / 0 |
| Q4 1024 base | 40.075860 / 39.176271 | 44.446231 / 46.568818 | 256 / 1 / 0 / 0 |
| Q4 512 LoRA | 9.967200 / 11.483013 | 13.062067 / 16.152563 | 258 / 16 / 2 / 0 |

LoRA 为真实 distill patch、strength1、238 applied projections（两臂相同），
258 calls 含2次 overflow retry，成功 block 仍256。runtime failure0。
condition/initial exact，完整8步轨迹、模板/二进制前后 unchanged。

**负结果保留**：512 两格更慢；1024 denoise 略短但 request-wall 更长。
这些带 dumps/observer 的单次运行不是 formal hot/reverse/load/memory
资格，不计算正式 speedup，不通过整体加速要求，不提升默认 raw 路由。

三格均720 raw misses、0 hits、714 evictions、15,925,248,000额外 source
read bytes；read host span 分别1.221698/1.654311/1.209382s。峰值
132,710,400 bytes（约126.6 MiB），完成时 live/entries 均0。六矩阵窗口
轮转不等于跨 step 的同矩阵复用；不要用8步乘次数估算“热命中”。
512 base 的 scale-cache 为42 hits/726 misses/714 evictions，与 affine
模型的672 hits/96 misses不同：raw allocation 换代会失去 scale metadata
复用。读流量和 cache churn 是下一步优化重点；窗口预算不是请求总 peak。

## 视觉和来源校验

| cell | final relL2 / cosine | 原尺寸 RGB SSIM |
| --- | --- | ---: |
| Q4 512 base | .1525485953 / .9884180773 | .9393023 |
| Q4 1024 base | .0661297572 / .9978146929 | .9859574 |
| Q4 512 LoRA | .1958856683 / .9808019070 | .9188188 |

三格旧 N1 fail、qualification_passed=false 不改。agent 原尺寸检查 whole
和全部同坐标三裁剪：1024整体非常接近，胸毛/雪粒有小变化；512 base
的耳部/胸毛/背景变化较可见；512 LoRA 的耳廓、头脸比例、眼部和背景
分支有明显局部改变，**不能据本轮观察视为“肉眼非常接近”验收通过**。
没有看到明显新棋盘格或破裂肢体，不意味着满足全部视觉要求。
自动 visual manifests 仍 pending，多 prompt/seed、盲测尚未完成。

两格 base 的候选 final latent payload、PNG 与此前普通 MLX affine 导入
Public Q4 候选分别 exact；这是这两个案例的跨保留构建结果对照，不是
全 decoder parity 或公平性能对照。LoRA 不借用旧 BF16 LoRA 质量结论。

质量工具新增 `gguf_raw_w8a8` 显式 route：绑定原 CPU-direct GPU label、
GPU 零 raw read、候选真实 raw read、256MiB/完成 drain、相同 source hash/
import plan/recipe/LoRA/library；普通 dense/Qwen/affine receipt 不能冒充。
仅在验证之后映射 canonical GPU label，输出保存原 `gguf_import`。

目录 `outputs/native-raw-gguf-ane-q4-{512-base,1024-base,512-lora-v2}-20261007/`。
按上表顺序 quality JSON SHA256：

```text
d09347a5752f051ba2994668b473ec3b1f3e6e85a348191528880ced9c948612
979ac50de2247c4d0477369b3b14b54357c180e2c5e6477a5b90eb5ffe48e506
16024ec021455dc2d77d45109e7f675fb608c4e2e67f59d92fa0f0d3c3e73602
```

第一次 LoRA runtime 命中 `select_acceleration` 残留 base-only gate，exit1
且无 PNG，保留在 `outputs/native-raw-gguf-ane-q4-512-lora-20261007/`。
修正后在新 v2 库重跑两臂，未拼接 v1 GPU 与 v2 runtime，也未覆盖失败。

## 回归、provenance 与后续

87项 host/tool tests通过；v1 新 raw FFN host/来源错误回退与 bank raw
窗口 actual Metal 测试分别通过；v2 14项 Public graph/Core ML/MLX/bank
回归通过，无 skip。bank 覆盖 cache identity、真实 raw bytes、escaped
lease/eviction/floor、取消、owner thread、source mutation 拒绝/clean refill。
FFN 覆盖 raw host graph、malformed geometry、完整 GPU recompute、
模型取源失败的 measured/untimed plan。
另7项 Private回归通过，无 skip：raw/affine/dense共享 stager、shared-event
ABA/cross-process cache、Comfy/direct staging、真实 W8 multichunk/
LoRA/weight switch/失败恢复，以及3项 native channel/prepared MLX 集成。
这仍不是 Private 整模型 raw 生成或其性能资格。
隔离 coremltools9 环境另5项 Public W8 exporter/真实 GPU→CoreML→GPU/
headroom/LoRA/失败恢复/corruption tests通过，无 skip。原有临时目录
ResourceWarning 保留，测试成功；这不是整仓绿色。

base 两格使用 v1 Public library SHA256
`90dbb60899363b33bfa037f844edf901fc829bf05444aa78f222d097a3f41823`，
build ID `tc-runtime-build-v1-33ca764d5a6d4ccd1ba35299638c1985ecc4c8734c95777d8da4d800618b7574`。
它已被 LoRA gate 修正的 current source 超越，不称其匹配 current tree。
LoRA v2 Public library SHA256
`fb434f1e5441f6ba4374fb905e560a781170def24b3c96508b8ed791017dfa68`，
build ID `tc-runtime-build-v1-22750a371e2549a06e9cdb08105221c12f5c3d468d09b8e48285111998d58f56`。
Private v2 library SHA256
`180a9a3ab1cba3f7bc8b266aaa743feda4e64499c436547e2bb79f9edc8c642b`，
build ID `tc-runtime-build-v1-4910ea9fc15c1b244f6519fd8748f704d229f283d869786e3b7af7737fda5dcd`。
两份 v2 均486 source inputs 独立匹配/seal一致；Public实际库 release guard
通过。保留构建包含既有未提交草稿，本轮只独立 stage 自己的修改；
owned-only Z translation unit 另做 syntax check，通过，不夹带草稿 commit。

下一步：实际预取 hit/等待/traffic 盈利测试、避免 raw 换代导致的 metadata
cache churn、Q8 与1024真 LoRA、Private整模型 raw、敏感层 GPU policy、
多 prompt/seed，以及完整 matched GPU 资格。不能因为这里速度失败就
把目标改成“只实现接口”，也不以支持兼容代替实际加速。
