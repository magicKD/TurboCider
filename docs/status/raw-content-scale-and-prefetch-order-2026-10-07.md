# Raw GGUF scale 内容代次缓存与 GPU-first 预取

2026-10-07，Asia/Singapore，M4 Max 64GB / macOS26.6.2。接续
[raw 来源窗口与负结果](raw-gguf-ane-source-window-2026-10-07.md)。
完整双后端、Z/Qwen base/真实 LoRA、512/1024、GGUF/ConvRot、视觉及
快于 matched optimized GPU 的目标仍 active；本页不是产品性能资格。

## 实现

raw bank 在完整读取且 verified SourceLease generation 检查成功后，为
每个 tensor 创建小的不可变内容代次 tag。物理 raw array 仍随六矩阵/
256MiB 窗口释放；tag 不包含 payload、ledger claim 或 source lease。
同 bank 同 tensor 的重填使用同 tag，新 bank/proof 使用不同 tag。
每次 raw read/cache hit 的原 source revalidation 和取消检查不改。

共享 stager 仅允许 immutable raw Q4_0/Q4_K/Q8_0/Q6_K 使用这个 tag，
scale-cache key 同时绑定 logical rows/columns/encoding 与完整 slice/
rotation/seed/basis recipe。物理 refill 的 buffer/offset/pitch 可以改变，
但 actual source extent/device/alias 仍先验证；Dense、affine、ConvRot 和
mutable source 不因内容 tag 获得宽松复用。A8 activation 不走该缓存。
cache 保持 weak identities、128 entries / 4MiB 上限，只缓存 row scales，
不缓存模型或 W8 codes。expired tag bypass，新的代次 miss。

Public/Private 的物理 prefetch identity 抽成同一
`ane_weight_identity.hpp`：prefetched **codes** 仍绑定 actual buffer/
pitch/offset、allocation/metadata/content generations 和 selection。允许
scale 跨 refill 复用，不允许把另一物理 producer 的旧 codes 直接激活。

另把 next-weight provider 调到 GPU complement 提交之后，再调用
optional prefetch。raw provider 的同步 CPU read 因而不再推迟 GPU
producer 的提交。row/channel、取消/异常 drain、whole-operation GPU
recompute、两套 W banks 与 shared-event timeline 保留。
`TURBOCIDER_RUNTIME_ANE_PREFETCH_AFTER_GPU=0/1` 保留同 binary 顺序
对照，default1；该设置进入 executor identity 和实际 receipt。
**prefetch 本身仍默认0**，不根据一次诊断开启默认预取。没有 native
INT8 MAC、FP32 ANE arithmetic、物理 GPU/ANE overlap 的 trace 声明。

## 先保留旧预取负结果

旧 raw v2 Public、512真 LoRA、prefetch1：248 submissions/hits、0失败，
720 raw hits/720 misses，PNG 与 prefetch0 exact。denoise11.801972s、
request16.458943s，仍慢于旧 prefetch0 的11.483013/16.152563s。
不能把“命中”当成盈利；读字节仍15,925,248,000，scale-cache仍42 hits/
726 misses/714 evictions。
目录 `outputs/native-raw-gguf-ane-q4-512-lora-v2-prefetch-20261007/`。

## 新同 binary 512 真 LoRA：缓存与顺序

Q4 original source、fox/seed42、8步、distill patch strength1、238 applied
projections。GPU/before/after/protected 四个独立进程串行、同 v2 Public
CLI/dylib，rows352、chunks1/fixed-async1、GPU IOSurface IO、prefetch1。
GPU control 为原 CPU-direct native affine consumer，没有 raw ANE reads。

| arm | denoise s | request-wall s | actual calls / retry / fallback |
| --- | ---: | ---: | --- |
| GPU | 10.009294 | 13.097692 | 0 / 0 / 0 |
| prefetch before GPU | 11.269879 | 16.000880 | 258 / 2 / 0 |
| prefetch after GPU | 10.450036 | 15.120502 | 258 / 2 / 0 |
| after + GPU blocks0,1,2 | 10.383253 | 15.047042 | 232 / 0 / 0 |

before/after 两臂均248 prefetch submissions/hits、0失败，scale-cache
96 misses / 672 hits / 0 evictions / 96 entries / 1,556,480 bytes，仍在
原4MiB界内。before/after 全部12份 tensor dump和PNG exact；after final latent payload/PNG 与
旧 raw v2 prefetch0候选也 exact。缓存实际消除了 raw allocation 换代
造成的 metadata churn，不改变 W8/A8 算法和精度。

约0.82s的单次 denoise 差异是同 binary **host scheduling 诊断**，不是
统计显著性、physical overlap 或正式 speedup。四臂都带 dump/observer，
没有 hot/reverse/strict load/memory 资格；候选仍慢于这个 GPU control，
更未证明快于全部 optimized GPU recipes。全部失败/不利时间保留。

## 敏感层 GPU 策略：改善但未达标

0,1是 noise refiners，2是首个 main block，旧 headroom overflow 两次均
来自 block2。显式保持这三层完整 GPU 后，24 forced-GPU blocks、232
Core ML calls，headroom1、retry0；scale-cache87 misses /609 hits、0
evictions，raw reads15,394,406,400 bytes。不是错误 fallback，也不更改
默认 layer policy；这些层不会 stage/调用 bridge。

512 after/protected final relL2 为 .1958856683 / .1624731001，cosine
.9808019070 / .9867898993，原尺寸 RGB SSIM .9188188 / .9387952。
旧 N1 fail 和 qualification_passed=false 保留。agent 原尺寸检查 whole
及三裁剪：protected 的耳廓/脸部比无保护候选接近一些，但眼部、毛发、
背景枝条与 GPU 仍有可见变化，不能视为“肉眼非常接近”验收通过。
没有看到明显新增棋盘格或断裂肢体，不等于所有图像合格；自动 review
仍 pending，多 prompt/seed 仍需补齐。

目录 `outputs/native-raw-content-scale-q4-512-lora-20261007/`。
after / protected quality JSON SHA256：

```text
cc282305128cb116bd56ce00dd78c32aa289c0886a99d03419264bd14c2003c1
45e10117318492e54ba9df22d75594e6399765ef1f04a1840d7448d95e3c51a4
```

## 1024 真实 Q4 LoRA 配对

同 v2 library、同 source/真实 distill patch，GPU/runtime各独立进程串行，
8步、238 applied projections、seed42。新增1056-row Public LoRA v2图，
仍是runtime weights/corrections/hidden ABI，没有学习到的模型/adapter const。
候选使用 after-GPU prefetch、GPU blocks0,1,2：24 forced GPU blocks、
232 Core ML/device-IO calls、headroom1、retry0、failure/fallback0，
scale-cache87 misses/609 hits；conditioning/initial exact、8步轨迹完整，
二进制/模板前后未变。

GPU/runtime denoise43.709351/42.558665s；request-wall48.151373/50.602330s。
**整请求候选仍更慢**，保留负结果，不能只挑 denoise 报正式加速。
这是同一单 prompt/seed、带dump/observer的诊断，不是完整资格。

final relL2 .0706766744、cosine .9975056320、RGB SSIM .9889266，旧N1
fail和qualification_passed=false保留。agent原尺寸检查whole和全部三
裁剪：主体、五官、姿态和光照很接近，胸毛、尾毛、雪粒和局部背景
略变，没有明显新增棋盘格/断裂肢体。该有限观察不替代多prompt/seed/
盲测或用户批准，自动visual manifest仍pending。

目录 `outputs/native-raw-content-scale-q4-1024-lora-20261007/`，quality SHA256
`317fd66d5117ab7d0ca45aa1d434071ae492e18ca354fbc314b93b77d3a3100c`。

## 回归与 provenance

15项 Public graph/Core ML/MLX/raw-window/shared-identity tests通过，无skip。
raw stager actual Metal test通过，涵盖四 raw encoding 的H128/H512、
unaligned refills、cache codes/scales/padding exact、weak tag lifetime/
new generation/mutable rejection，原9 encoding/dtype与finite/refill
控制仍通过。另6项 Private shared-event/Comfy/W8 multichunk/LoRA/
失败恢复/native channel/prepared MLX tests通过，无skip；隔离
coremltools9下5项 Public W8 exporter/device-IO/失败恢复 tests通过。
另87项相关host/tool tests通过。不是全仓或 Private整模型 raw 性能验证。

v2 Public library SHA256
`0e694af1267e993492ea7ff8767e202f53f0f19176605455b2f7488d3b469870`，
build ID `tc-runtime-build-v1-2eef07a33b187e43285c9162270d23e20a773a90c93f60225a56195b92db571b`。
v2 Private library SHA256
`2644170665d11b59216c688632238cfd38c24527dda4a922b8b3879b6a59431c`，
build ID `tc-runtime-build-v1-4a829fe5187e9884cff7a29450caca32eb0723dae381e035a9c6323a0f6a9aad`。
两库487 source inputs独立匹配/seal一致；Public实际库 release guard通过。
保留构建含原草稿，commit只包含本轮独立 stage 的实现，不夹带草稿。

后续仍要避免 raw 重读、实测 resident retention 和更合适的来源/后端/
partition policy、更多图像/分辨率及正式 matched GPU。兼容/命中/数值
改善不能代替完整目标，也不因本轮负性能结果缩小目标。
