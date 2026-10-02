# 24 · 固定 GPU bank、dependency-ready 与独立 source block 验证

[目录](README.md) · [23](23-raw-gpu-affine-streaming.md) · [冻结验收](11-acceptance-profiles-and-feasibility.md)

本阶段实现并验证固定 GPU 输出 bank、源浮点 refiner 常驻、GPU packing 的显式 lazy
依赖，以及 bounded raw 路径的独立逐 block source 验证。默认、普通构建和生产 catalog
不变。最终同库 Q8/Q4 比最快已测 BF16 慢 **31.59% / 28.38%**，仍未达到≤20%目标；
不能把本阶段的机制/数值验收或6GB observed fit称作完整加速交付，完整目标继续 active。

## 1. 固定资源和内容 ticket

`gguf_gpu_affine_fixed.{hpp,cpp}` 在 owner 创建 pool 时一次分配每个 projection 的
codes/scales/biases 和独立4-byte error word；reservation先于分配，claim由 array Data
持有。当前只有一个 GPU 输出 bank，raw p0/1/2仍为1/2/3个 CPU raw槽，不按层常驻
完整 affine/dense 模型。逃逸 view保留原 claim，不能只删除 map 就说资源已经释放。

新增显式 profiles：

- `z-raw-gpu-fixed-f16-v1`：同步 packing、流式 source-float refiners。
- `z-raw-gpu-fixed-refresident-f16-v1`：同步 packing、约1.455GB浮点固定/refiner字段常驻。
- `z-raw-gpu-dependency-refresident-f16-v1`：同一固定容量和 refiner政策，packing是
  参数图的显式依赖，不在每层 QMM 前额外 host等 packing完成。

都要求 experimental build、GPU、packed_streamed、compile_gpu和明确近似授权。
仍只支持rank2完整Q4_0/Q4_1/Q8_0 GPU packing；K/IQ不静默切换 CPU/GPU后端。
新的 refiner政策要求这些非主层字段确为 source floating，不重新量化它们。

pool记录 GPU内容 ticket；即使下一 raw槽已填好，也拒绝在旧 ticket退休前重写 GPU
bank。retire必须匹配pool/slot/pass/content身份，并覆盖实际最后reader完成；错误 ticket
拒绝。下一ticket生成新参数身份、复用同一物理 backing，A→B→A不捕获旧权重常量。

fixed profiles的 direct pread取消边界为8MiB；worker仍只做CPU读取/小字段转换，
不创建MLX对象、不提交GPU。BF16→FP16 importer alias改为一次原地SIMD转换，
保留旧loader数值，不称作保留源BF16计算精度；有限值/FP16 overflow仍失败。

## 2. 不把 dependency-ready 伪装成 validated Ready

CPU Ready仍是raw GGUF。新`FixedAffineDependency`返回显式依赖的多输出节点：

1. 所有几何、alias和容量先检查；prepare只建图，不dispatch/不声称解码完成。
2. `eval_gpu`把输出alias到已admit bank；GPU清error word，再GPU packing。
3. 每个QMM消费该节点的codes/scales/biases；MLX依赖/同stream资源hazard保证顺序。
4. owner等待整层最后QMM reader，检查全部error words、拆primitive/sibling对raw的引用，
   再retire bank和发布层结果。坏metadata允许内部计算被提交，但绝不发布成功层/PNG。

primitive不把两个不同内容ticket判为equivalent；无额外affine输出分配，没有dense权重。
失败后的GPU status通过GPU清零，干净重试不能继承旧错误。
`gpu_prepare_wall_seconds`在此profile仅表示host建依赖图的时间，新增
`gpu_prepare_timing_scope=host-dependency-construction-only-v1`明确这一点：约.0018s不代表
packing的GPU工作消失或仅用了.0018s。其GPU成本仍在denoise/request时间内。

## 3. 真实数值、生命周期和普通构建门禁

最终experimental library SHA-256：
`e75ec53c4e769dda5d9175da337fa729da2619a8fdc45b6f9a3b1887b71cd145`。

- Metal component：27几何，三type、rows1/7/33、K32/64/256，CPU独立packing逐byte
  exact；同步/dependency fixed A/B/A、实际QMM consumer、物理backing identity、
  escaped/lazy claim、nonfinite/overflow拒绝及clean retry通过。
- 真StageExecutor：dynamic/fixed/dependency三模式、K1/K2/K3、两次pass；8fills/readers、
 无source重读/隐藏CPU repack、busy拒绝、错retire拒绝、下一ticket内容与释放归零通过。
- CPU alias exhaustive65,536 BF16 bit patterns，scalar/SIMD、tail、finite/overflow、取消通过。
- 最终host/parser/measurement/screen/storage矩阵30项通过；Metal/parser/普通generate
  gates矩阵10项通过。普通库11条显式route case（包含新fixed/resident/dependency）拒绝。
- 额外原CPU directory/decode矩阵11项通过、13项可选oracle/model fixture skip；这些
  skip没有计入本阶段通过，更不能声称全Q4–Q8 oracle qualification。

初次dependency escaped-claim测试在全局drain前要求只剩一个claim失败。增加实际
`mx::synchronize()`排空framework完成引用后才检查唯一物理claim，最终通过；未将
未drain引用标成已释放，也没有减少应保留的claim。沙箱中无Metal设备的失败不计为硬件
测试通过，随后实际获得设备访问再跑。Q4最初误用不存在的fixture目录被组件binding
拒绝，原失败目录保留；使用真实`q4-main`的新运行，没有覆盖失败结果。

## 4. bounded raw 的独立137项 source 验证

`--validate-source-blocks`现在可用于显式raw GPU诊断，timing/memory和环境污染仍拒绝。
reference使用另一个CPU-affine有界pager、独立slot/filler和原compiled packed算术；
从相同noise/conditioning独立Euler演进，不把candidate input借给reference，也不调用
被测GPU packer来自证。每pass只保留34个activation outputs，不建resident dense模型。
streamed refiner/main observer都在实际reader和status验证后调用。

| 512²/portrait/seed42/4steps、p2 | 136个block N1 | 最大block relL2 | 最终latent |
| --- | --- | ---: | --- |
| Q8 dependency fixed | 全部通过 | .00504138745 | N1通过 |
| Q4 dependency fixed | 全部通过 | .00522184794 | N1通过 |

两条独立reference的最终latent文件分别与22对应reference字节exact；candidate的
conditioning、initial/every-step/final latent、decoded和PNG与22已验收resident FP16
consumer exact。Q8 p1/block5取消status2、无PNG，同engine retry status0，所有端点exact；
Q4 p1两请求端点exact。不是48-case媒体、其他shape/prompt/seeds或生产质量认证。

## 5. 同库独立速度/内存：仍失败

M4 Max64GiB/macOS26.6.2/MLX0.32.0；原BF16 encoder/VAE/tokenizer内容相同、34valid
tokens/combined1088。GPU tuning沿用21/22较快BF16；每arm4个热样本、1个保留warmup，
内存另进程10ms采样。无observer/source-validation混入计时。

| arm | 热wall median | denoise median | 观测process peak bytes | 对BF16 wall / memory |
| --- | ---: | ---: | ---: | --- |
| 纯GPU BF16 | 3.736814s | 3.497756s | 23,012,435,880 | 1 / 1 |
| Q8 dependency p1/cache1GiB | 4.917347s | 4.291904s | 9,399,409,264 | 1.315920 / .408449 |
| Q4 dependency p1/cache1GiB | 4.797168s | 4.177218s | 9,400,212,152 | 1.283759 / .408484 |

p2对应Q8/Q4热median4.946890/4.816509s，没有比p1更快，不把增加slots说成持续供给
加速。原encoder约9.4GB的全请求peak主导两条记录；这不是两种DiT权重压缩比例相同。
最高采样gap<16.0ms、swapout0。当前profile不能据此称10GB扣10%余量9GB已满足。
正式ABBA/CI/p90/cold/全部cells与driver连续时间upper仍未完成。

[Q8 BF16负screen](validation/fixed-dependency-q8-bf16-screen-20261002.json) ·
[Q4 BF16负screen](validation/fixed-dependency-q4-bf16-screen-20261002.json) ·
[hash绑定诊断、逐block和取消](validation/fixed-dependency-gpu-progress-20261002.json)。

## 6. 真实逐tensor预测，不直接把Q4说成整个进程30%

`tools/validation/gguf_storage_estimate.cpp`使用生产checked directory reader，无payload
decode/mmap/GPU。输出每type的elements、源bytes、全BF16逻辑等价bytes、native affine
codes/scales/biases与16KiB字段capacity upper；K等未有affine consumer时输出null，
不把未知项当0。mixed fixture、对齐和截断拒绝通过，真实两文件hash与最终screen绑定。

| 全453tensor模型的理论weights（不含其他组件/activations） | Q8 | Q4 |
| --- | ---: | ---: |
| 全BF16逻辑等价 | 12,309,817,472 | 12,309,817,472 |
| GGUF源payload | 7,224,676,608（58.69%） | 4,509,395,072（36.63%） |
| resident affine payload含原浮点字段 | 7,563,825,408（61.45%） | 4,848,543,872（39.39%） |

单量化矩阵Q8_0源34/32 bytes、affine36/32，分别为BF16的53.125%/56.25%；Q4_0
源18/32、affine20/32，分别28.125%/31.25%。这里约1.45GB浮点refiner/固定字段没有
变成Q4，因此mixed weights为39.39%，不是30%。streamed实际活跃峰值则是fixed fields、
raw slots和一个GPU bank之和，不应套用整个resident weights预测。完整请求另有encoder、
VAE、cache/activations及未知driver资源，process比值不能由checkpoint比值单独推导。

[完整type账本](validation/fixed-dependency-storage-prediction-20261002.json)。复现：

```sh
xcrun clang++ -std=c++20 -O2 -Wall -Wextra -Werror -I native/core \
  tools/validation/gguf_storage_estimate.cpp -o build/quantized-execution/gguf-storage-estimate
build/quantized-execution/gguf-storage-estimate models/z-image-runtime-gguf-q8/z_image_turbo-Q8_0.gguf
```

## 7. 6/8/10/16 GB容量screen与下一步

另进程使用已有bounded Qwen3 Q8 encoder（streamed p2、1GiB managed ceiling）与Q8
dependency p1/cache1GiB组合：全请求观测peak **4,895,622,440 bytes**、managed DiT
peak2,044,682,516bytes、gap15.59ms、swapout0；6/8/10/16 GB及GiB扣10%余量均为
**observed fit only**，没有真实小机器/whole-request hard-cap资格。
该组encoder内容/conditioning与BF16不同，不混入第5节同组件比较。热median5.292460s，
仍不能声称达到BF16速度目标。

[GB容量screen](validation/fixed-dependency-q8-budget-gb-screen-20261002.json) ·
[GiB容量screen](validation/fixed-dependency-q8-budget-gib-screen-20261002.json)。

下一轮优化必须覆盖完整request，而非只继续降低packing的host统计：

- bounded encoder即使cache-hit，text阶段仍约.35s：`conditioning()`在检查cache前每次
  重建Qwen3 GGUF encoder、目录/config和tokenizer验证。应复用**verified metadata only**，
  保持source/config/tokenizer重绑定和generation检查，不跨请求保留dense weights。
- raw路径VAE约.39s，对应BF16约.22s；source-float加载约.18s。阶段cache/固定字段
  生命周期值得独立profile优化，但retention必须重新分账，不能让6GB VAE峰值越界。
- 同预算比较已有较快resident FP16 packed Q4和bounded encoder组合；不能只优化低内存
  raw fallback而忽视可能更快且仍能fit8/10GB的路线。16GB候选仍保留22，未被本阶段替代。
- single bank每层实际reader边界、raw→affine额外GPU流量仍需算子/调度证据；
  exposed ready wait本cell很小，不能只凭约1.3s后台read就认定SSD是全部瓶颈。
- tiles/required-site闭包、更多格式/组件、48-case媒体、ConvRot旋转域/非W8A8低内存、
  ANE W8A8完整request快20–30%、M5静态backend合同仍在原完整目标内，未据此标记完成。

最终模型命令沿用runner及上述GPU tuning。例如新raw正确性诊断：

```sh
TURBOCIDER_Z_MPP_SWIGLU=1 TURBOCIDER_Z_MPP_PROJECTIONS=1 \
TURBOCIDER_Z_MPP_QKV_PREPARE=1 TURBOCIDER_Z_GATE_NORM_VIRTUAL_THREADS=128 \
TURBOCIDER_Z_CACHE_CONTEXT=1 .venv/bin/python tools/native/run_z_image_gguf_quantized.py \
  --library build/quantized-execution/libturbocider.dylib --model models/z-image-runtime-gguf-q8 \
  --output outputs/quantized-execution-fixed-gpu/q8-new-validation \
  --prefetch 2 --source-residency packed_streamed \
  --precision z-raw-gpu-dependency-refresident-f16-v1 --raw-gpu-cache-bytes 1073741824 \
  --runs 1 --validate-source-blocks --dump --bind-components
```
