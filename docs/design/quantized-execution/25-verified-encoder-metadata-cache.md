# 25 · 仅复用 verified encoder metadata，继续降低完整请求开销

[目录](README.md) · [24](24-fixed-gpu-bank-and-dependency-ready.md)

本阶段消除了Qwen3 GGUF prompt-cache-hit前重复解析/验证tokenizer的主要CPU开销。
它是**metadata/tokenizer复用**，不是跨请求保留encoder weights或dense graph；没有新增
量化近似，也没有授予6/8GB速度、完整内存、媒体、ConvRot或W8A8资格。完整目标仍active。

## 1. 实现和失效规则

`Qwen3GgufPreparedSource`持有verified SourceLease、config和CPU tokenizer，不持有
Tensor、fill worker、GPU graph或engine取消引用。每个真正encode仍创建独立ledger、
source/slots、math状态和executor，完成后drain/release。SourceLease的文件generation可
跨请求稳定，但新encoder有独立单调request generation，不把source generation误当成
每次执行身份。结果的`conditioning_producer_generation`在prompt cache hit时仍标识原
conditioning producer，而不是虚报又执行了35个layers。

prepared source复用先核对当前指定的三条named paths，再检查held fd、mtime/ctime、
inode和symlink target。变更时新请求重新prepare并执行全部content/config/tokenizer
校验；不把旧metadata授权给新字节。cached-conditioning identity继续包含source/config/
tokenizer hashes及prefetch、managed ceiling、decode policy、source residency。
cache hit也调用policy验证，不能绕过`DEFER_LAYER_EVAL`/interval冲突或容量/布局变化。

request保留轻量source lease并在PNG publish前再检查；encoder结束后、DiT过程中修改
配置也会失败且无PNG。新prompt/binding encode失败不会遗留缺producer metrics的旧
conditioning cache。generate结果tokens来自成功encode的实际tokens，避免重新使用旧tokenizer
对象推导新conditioning的rows。完整CPU metadata heap/driver upper仍是unknown。

显式GGUF encoder实验默认启用metadata复用；原BF16模型/原encoder算术不变。对照开关：

```text
TURBOCIDER_QWEN3_GGUF_METADATA_CACHE=0|1
runner --encoder-metadata-cache off|on
```

不绑定GGUF encoder却提供该control会拒绝。`off`每请求重建metadata，保留数值相同
的对照。report新增`source_metadata_policy/reused/preparations`，memory screen v2
核对实际policy与control，拒绝缺失、伪报reuse、非整数generation或改变lifecycle。
runner的diagnostic alternate-prompt也允许bound encoder A/B/A，timing仍拒绝。

## 2. 验证

最终experimental library：
`6b042a452a8f2e43e73fff4e06326f1d4a6395ad4d1b655e413a0ba9a70e7ee7`。

- prepared-source实机组件：GPU active bytes不增加；tokenizer A/B/A、unchanged reuse、
  execution identity、p3/未知residency/interval冲突、cancel、新config和symlink改指向
  拒绝旧proof，malformed replacement仍拒绝，全部通过。
- 实际Z engine、512²/1step：warm policy错误无PNG；临时config加空格后重新验证/重编码，
  PNG exact；prefetch变更重新encode但复用metadata；取消encoder layer5后同engine
  retry成功；A/B/A回到A PNG exact；DiT block2修改临时config导致请求失败/无PNG，
  下请求重新绑定成功/PNG exact。原模型文件未修改。
- Q8 raw profile两次真实512²/4steps，PNG与24同组件控制exact；cold preparations=1、
  warm仍1且reused=true，conditioning producer generation没有伪增。
- resident Q4 + Q8 encoder（额外conditioning源量化保持单列）独立136个block+最终
  latent N1全部通过；其全部diagnostic endpoints与同库metadata-off控制exact。
- host/parser/measurement/screen/storage矩阵31项通过；Metal/metadata/parser/普通
  generate门禁矩阵11项通过；上述真实engine故障矩阵1项通过。普通build也拒绝显式
  Qwen3 GGUF metadata-cache路径；实验scope未泄露到普通generate。

[hash绑定数值诊断](validation/metadata-cache-diagnostics-20261002.json) ·
[测试/速度回执](validation/metadata-cache-progress-20261002.json)。均不是48-case媒体资格。

## 3. 同库on/off、独立速度和内存

Q8 DiT仍为24的dependency p1/refresident/cache1GiB；encoder为bounded Qwen3 Q8、
streamed p2/1GiB managed ceiling。相同source/组件/GPU tuning/portrait/seed42/4steps，
每arm4个cache-hit热样本+1保留warmup；官方计时无observer，内存另进程10ms采样。

| metadata policy | 全请求warm median | text阶段median |
| --- | ---: | ---: |
| off，每次重建 | 5.284996s | .341652s |
| on，仅CPU metadata复用 | 4.928479s | .000226s |

全请求ratio=.932542，本地screen少6.75%时间，PNG exact；不是正式ABBA/CI/cold资格。
本轮没有将不同encoder的结果混入同组件BF16 screen，也没有授予BF16≤20%目标。
24已证明低内存raw路线仍超速度硬门，本阶段只是删除其中一项真实开销。

on独立完整请求process peak **4,928,226,720 bytes**、swapout0，6/8/10/16 GB及GiB
扣10%余量仍均observed fit，非强制RAM cap/真实小机器认证。
[GB screen](validation/metadata-cache-q8-budget-gb-screen-20261002.json) ·
[GiB screen](validation/metadata-cache-q8-budget-gib-screen-20261002.json)。

## 4. 保留resident Q4负结果，下一步不再误归因

较快22候选的4.4s是**保留完整packed bank**的warm路径，不是每次低内存重新导入的
速度。新测未保留bank、VAE前verified释放的resident Q4 + bounded encoder：

- 独立memory run peak **7,803,327,000 bytes**，超过8GB可用7.2GB及8GiB可用
  7,730,941,132bytes；可observed-fit于10/16GB，不宣称8GB通过。
- metadata-off诊断warm wall约6.981s，其中import **2.219246s**，decode/repack
  **2.025042s**、实际source read仅.169379s。on的memory诊断warm约6.626s。
  这些diagnostic/observer时间不能当正式perf samples，但足以拒绝“这条路径已经更快”。
- 现有`pack_native_affine`对codes/scales/biases三次遍历，每group标量half转换和Q4
  nibble重排，正是后续优先优化对象；不是把这2s全算为SSD等待。保留负结果，不删除
  慢样本，也不把16GB retained成绩嫁接给8/10GB release-before-VAE。

下一阶段可做lossless fused/SIMD affine packing（原codes/scales/biases bit-exact oracle，
特别穷举FP16 bias的normal/subnormal/±0/overflow），再独立测阶段cache、低内存runtime
与更快GPU kernels。不能通过跨请求保留完整dense模型或扩大未声明内存来凑速度。
6/8/10GB的≤20%目标、更多shape/组件/IQ与48-case媒体、整体allocation/envelope闭包、
ConvRot非W8A8低内存、GPU/ANE W8A8、M5静态backend仍全部保留在原目标内。
