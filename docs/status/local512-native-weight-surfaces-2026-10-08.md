# Native W8 surface binding：去掉缓存复制，但收益仍依 workload

2026-10-08，Asia/Singapore，M4 Max64GB / macOS26.6.2。接续
[复制型 codes cache](local512-weight-code-cache-2026-10-08.md)。完整
Z/Qwen base/真实LoRA、1–2参考图、encoder、GGUF/ConvRot及盈利目标仍
active；只使用本地模型/adapter，无下载、模型改写或磁盘权重副本。

## 实现

显式 `TURBOCIDER_RUNTIME_ANE_WEIGHT_CODE_CACHE_MODE=surface`，预算仍由
`TURBOCIDER_RUNTIME_ANE_WEIGHT_CODE_CACHE_BYTES` 控制，默认0；默认
storage仍copy。mode进入executor identity。Shared Device的`bind_w8`
经原全部source/metadata/device/extent/basis/alias验证后，才选择缓存。
hit返回已经GPU完成且通过finite/range检查的immutable IOSurface，
不再blit W codes/scales。miss直接在新缓存surface内执行原stager；
未准入则原fixed scratch路径，数学、精度与原codes不变。

Private/Public两种W8 graph的两执行bank分别保留自己的fixed scratch
和当前bound references。每次refill先将**退役的目标执行槽**reset到
原scratch；当前reader/future槽不能被改写。program input与down restore
scale使用bound references，不再默认读取scratch。晚期失败仍重算整个
operation，不发布半成品；backend selector/默认route不变。

容量claim绑定到Surface backing：cache entry删除、producer结束或
当前bank引用释放都不足以提前归还尚有row-view/reader的容量。计算
IOSurface physical pitch/page upper，包含row scales的64-byte pitch。
新staging/upload拒绝用cached immutable surface作写目标；caller不能
把上次返回的cached binding直接当下一次fallback scratch。claim存活
到最后backing alias释放，仍保留原4GiB system reserve/2GiB optional
tier，未扩大预算或把它冒称进程/driver RAM cap。

回执分开`copy_hits_session_total`与`surface_bind_hits_session_total`，
其和覆盖总hits；native storage、ready/retained/live/peak及declines
独立报告。直接binding是已验证reader选择，不是新GPUcopy或物理
GPU/ANE overlap证明。fixed-target API保留copy语义作为对照。

## Qwen512² LoRA一/两图：真正零复制，整请求仍慢

同库Viggle r256、六步/seed29、三个fresh prompts、原有序reference；
GPU encoder、Private DiT5120、shared gate/up ranks，相同encoder source
retention。surface预算1280MiB，off预算0。冷请求单列，一冷两热、
100ms process-tree采样；无continuous-load资格，保持diagnostic。

| workload / order | GPU warm median s | hybrid off s | hybrid surface s |
| --- | ---: | ---: | ---: |
| 一图：surface→off→GPU | 10.977447 | 11.825826 | 12.178534 |
| 两图：GPU→off→surface | 13.144468 | 14.411138 | 14.628537 |

两种workload各on/off三张PNG exact，也与此前同Private recipe结果exact。
227实际adapter projections、每请求192个hybrid blocks、384个shared
rank arrays；一/两图224/256calls/request。failure/fallback/retry0，
source仅load1次。surface最终63ready entries、fills63、累计1071个
direct bindings、copy hits0；actual live/peak1,340,473,344 bytes，未超
配置1,342,177,280。不是marker假称零复制。

6个memory报告complete、系统swap-in/out0；一图surface/off/GPU peak约
39.6/38.3/40.3GB，两图GPU/off/surface约40.8/38.9/40.2GB，scope为
load/cold/warm/exit进程树，不含外部服务/driver归因。两种workload顺序
相反不等于同一workload的双向重复，不合并成更有利的分母。
零复制没有带来当前Qwen盈利，不能仅凭这个结果断言surface IDs、driver
binding、内存pressure或ANE prediction哪个是唯一原因；需要独立定位。

目录 `outputs/local512-qwen-edit{1,2}-weight-surface-v1-diagnostic-20261008/`。

## Z512 base：混合路径约1.23×，不是新cache带来1.23×

本地BF16、fox/seed42、8步、Fa4096、fixed-async/F32 join、prefetch0，
同一库、原完整优化GPU控制，GPU/runtime各一冷两热。

| window / order | GPU warm median s | Private warm median s |
| --- | ---: | ---: |
| surface，GPU→Private | 6.995911 | 5.665270 |
| cache off，Private→GPU | 6.989535 | 5.685262 |

缓存on/off的Private PNG全部exact，两个GPU arm也exact。surface实际
83ready entries、1909个累计direct bindings、copy0、failure/fallback0，
每请求256calls。4个memory报告complete，swap-in/out0；surface GPU/
Private peak约22.95/24.07GB，off Private/GPU约22.74/22.95GB。

目视原尺寸whole和全部三同坐标裁剪：狐的姿态、耳/脸、毛色、眼鼻、
尾巴与构图很接近，胸毛/雪粒/背景枝条细节略变，未见新断裂或色块。
仅一个scene/seed的有限agent观察，不是用户批准或多seed资格。自动
visual manifest保持pending，原数值失败记录不改写。

新cache自身只有约0.02s差异且顺序不同，不能宣称其独立显著收益。
原Private混合对GPU的有限诊断较快；不把这个已有收益全归新cache。
两window均没有strict load资格，也不证明物理overlap。

随后strict窗口保留所有gate：GPU四请求完成，54个load samples中2个
external inference CPU busy，被拒绝，Private未开始，summary仍
incomplete/accepted trials0。不能拼接诊断Private、删除busy samples、
停止外部进程或将此strict失败追认成正式1.2×。

目录 `outputs/local512-z-base-weight-surface-v1{,-off}-diagnostic-20261008/`；
strict失败为 `outputs/local512-z-base-weight-surface-v1-strict-20261008/`。

## 验证、工具、接续

共享Metal测试22种format/storage case exact，覆盖dense三dtype、affine
Q4/Q8、raw Q4_0/Q4_K/Q8_0/Q6_K、Comfy raw/packed、A/B/A、physical
refill、padding、generation expiry、finite/failure/refill。另独立检查
native首次直接产出、hit同IOSurface、copy/bind计数、immutable target
拒绝、hit仍检查physical span、cache clear后view claim、独立scratch
和leased预算拒绝。Private selected19项、另actual native-cache channel
1项通过；Public selected18项、已有隔离coremltools9实际Public W8
5项通过；screen/tool73项通过，无skip，不是全仓/全模型矩阵通过。

Qwen screen新增storage选择及实际copy/bind完整计数校验。通用model
screen可显式给runtime budget/storage、完整GPU不加缓存；Z/GGUF/Qwen
均能用同一共享stager。Public actual release-binary guard通过，两库各
491source inputs匹配。构建含既有ConvRot草稿，是working-tree snapshot，
提交只含owned hunks，不冒称clean staged-tree rebuild。

```text
Private cf02da5367f74e0c5d1bb795b2e13046aa70e0417dc616f1de584da71f81ce54
Public  26419067a4ada38f3023075202f42d3887c99730625444e6c9fb57b9a8aac94c
```

所有owned jobs terminal后清理两build的423个`.o`，32,964,424 logical
bytes（约31.4MiB）与两空module-cache，可重建。保留库/CLI、日志、
manifest/PNG、模型与adapter，无大activation dump。机器记录见
[本轮证据](../design/validation/local512-native-weight-surfaces-20261008.json)。

接续仍需Qwen真实盈利的分区/driver绑定/LoRA handoff、Z真实LoRA、
Qwen base generation及encoder盈利、Public实模型、GGUF/raw读取与
ConvRot融合、多prompt/seed视觉和有效严格窗口。cache默认继续关闭；
不能以这些接口/兼容测试代替完整加速目标。
