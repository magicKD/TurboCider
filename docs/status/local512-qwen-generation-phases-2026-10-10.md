# Qwen512 生图分阶段：base 后续 FFN 并行有收益，六步 LoRA 不套用

2026-10-10，Asia/Singapore，M4 Max64GB/macOS26.6.2。接续
[首步逐层选路](local512-qwen-prefill-layers-2026-10-09.md)及
[共享GPU kernel](local512-affine-shared-word-2026-10-10.md)。本轮补齐最新
本地Qwen Image2.1的512² base与原六步LoRA生图完整对照；未下载/改写
模型、adapter或ref，原用户草稿保留。完整目标仍active，以下均为诊断。

## 实现：同一分阶段机制也用于没有参考图的生成

既有 `TURBOCIDER_QWEN21_RUNTIME_FFN_PHASE=all|prefill|decode` 的非默认
scope现在允许原resident512 `image.generate`且inputs为空，原1–2ref512
编辑资格不变。仍要求明确approximation、固定正Private channels、GPU
encoder、无streaming/预算/PE和其他时间复用。默认all、算术/precision/
source/finite/shape/memory/完整GPU恢复均不变，无新kernel/权重表示。
旧early/late identity和actual-prefix-state hooks继续隔离策略；未选中的
阶段是原完整compiled GPU，而不是zero-channel split或额外timing probe。
前/后层GPU的实验列表仍edit-only，不让它随生图scope自动放宽。

`qwen_ffn_phase_screen.py --generation` 不要求ref，拒绝generation与ref/
edit-layer组合；旧编辑参数继续兼容。wrapper新 `--phases` 记录enclosing
CPU observation，始终qualification=false，原strict-load机制不改。
生图不再带编辑专用LoRA ref512 flag。新48-case compiled fixture覆盖
base/真实LoRA、with/without ref、prefill/decode/all、prefix命中/变化及
disabled phase无stage/probe；原18-case按层fixture保留。

## 同最新库：base40与原LoRA6，不跨两种schedule借分母

原本地BF16 DiT/Qwen3-VL/VAE，512²、seed29、三个不同fox prompts（natural/
soft/warm winter light），conditioning全miss。每臂独立process、一冷两
fresh warm；同GPU eager encoder和admitted retained original source。
Private Fa5120/Fg7168、c1056 v2、W8A8/F32 partial join、fixed async1/
stage/fence1，prefetch/lookahead/cache/time reuse/down-rank split关闭。
LoRA为原Viggle v0.2.1 r256/strength1、inference-time/227bindings，所有臂
同joint BF16 A/B/F32 ranks，hybrid共享原gate/up input ranks。

| route | base40 warm request s | base首步 / 后39步 s | LoRA6 warm request s | LoRA首步 / 后5步 s |
| --- | ---: | --- | ---: | --- |
| complete GPU | 43.324888 | 1.124465 / 40.920705 | 8.603859 | 1.263297 / 5.860550 |
| only prefill parallel | 43.293465 | 1.098605 / 40.894301 | 8.507092 | 1.289455 / 5.869694 |
| only KV-hit parallel | 34.759631 | 1.148878 / 32.294058 | 8.531077 | 1.288396 / 5.923432 |
| both phases parallel | 34.653947 | 1.085971 / 32.278767 | 8.594763 | 1.274210 / 5.929769 |

base order GPU→prefill→decode→all；LoRA相反order，不是同workload ABBA。
两个enclosing CPU-load checks均fail，不称稳定正式倍率/设备独占；保留
所有cold/warm/prompts/观察，不与旧库或编辑速度相乘，不拼各臂最佳stage。

base全阶段相对同窗口GPU名义request减少20.01%，KV-hit-only减少19.77%；
prefill-only仅.073%，不足以成为明显整图优化。首步本来只有约1.12s，
后39步是主要工作；KV cache减少文本前缀，但目标图attention/FFN仍在算。
因此不能把“后续不值得并行”的编辑判断套给base40生图。

LoRA6四路线8.51–8.60s。prefill-only名义少1.12%，但首步/后五步都没有
变快，且两warm整请求波动约.5–.86s，不能把wall微小差值全算给ANE。
all仅少.106%，decode-only少.846%；没有可靠新phase赢家，继续优先完整
GPU，不将base20%或多参考编辑首步收益套给六步LoRA生成。
需要同workload更多样本/反序确认；这不是LoRA数学未执行的降级路径。

真实first rows1055、后续1024，均一chunk。base每请求calls：prefill32、
decode1248、all1280；LoRA分别32/160/192。request-local phase与累计
3请求counter一致，disabled phase calls0；zero failure/fallback/overflow
retry、headroom1。LoRA所有臂227bindings，实际成功shared rank blocks
匹配选中阶段。native source proof各hybrid cold full hash读取
1,359,147,904bytes，warm hit1/read0/rebindfalse；不删校验。

encoder本轮全GPU。此前完整 [compiled/native-GQA/Private筛选](local512-encoder-compiled-gqa-2026-10-09.md)
未建立显著整请求收益，故不盲组合。source retention是同条件控制，不把
DiT20%说成encoder盈利；视觉塔/语言/DiT仍需独立按实际cost选择。

## 保留失败、图像与内存

第一轮native plan fixture将编辑的reference_size512带到无ref生成，被
原resize guard正确拒绝；修正为1024，未放宽native guard，完整9项重跑
通过。首个LoRA generation runner又带ref512 edit-only env，native在
模型工作前拒绝；summary保持incomplete。新pure helper只在确有ref时
发出flag，加unit检查，另开v2目录完成；旧日志/观察/失败不覆盖。

已看每workload四个route的case1 whole512：狐姿态/脸/耳/眼鼻/尾巴、
毛色、雪/枝条布局非常接近，凝视/毛纹/雪粒/细节有小变化，未见新增
明显棋盘格、断裂或色块。仅有限agent观察，不是所有scene/seed/detail
或用户批准，不声称byte-equivalent，不用严格latent作为图像接受门槛。

8个memory reports complete，最大process-tree phys-footprint
39,892,936,912bytes。base prefill-only/decode-only分别system swap-in
65,536bytes，其余0；全部swap-out0。compression约8.47–15.20GB（trial
累计计数），不能说全窗口无swap/compression或归因特定kernel/外部App。
scope为100ms进程树load+cold+warm+exit，不是整机RAM硬上限或device trace。

## 构建、回归、清理与下一步

最终Private9/Public8 selected native tests、host9项pass无skip：新48个
compiled generation/edit phase cases、原18个GPU-layer cases、joint math、
held-fd source、typed lifetime/source/memory及实际Private晚期全GPU恢复。
Public actual release class/flags/links guard exit0。不是全仓或完整全模型
matrix资格。两native-only build exit0、各502source hashes与当前tree
独立匹配/seal一致，含原用户ConvRot草稿，不是clean staged-only/App发行。

```text
Private b664b504e60ebde3dd2fc1e74dc176319a769fb3d3c549fa8ac47bb25fbe1bf0
Public  a5a147a94eafc49e6214ccc6c517ca49cdceef402d52146a0c9bea4d557682b3
```

确认所有owned jobs terminal后，清理两build的429个可重建`.o`、
33,488,344 logical bytes（约31.9MiB）及两个空module-cache，回查objects0。
保留全部库/CLI/probe、raw/失败logs、PNGs/manifests及原模型/adapter/ref，
不清用户cache、不向外部process发信号；对象可按原build命令重建。

建议仍按任务区分：base40生图可用all-phase Private候选；六步LoRA生图
优先GPU；1–2ref编辑按已测首步并行/后续GPU独立选择；encoder保留更快
GPU。未比较当前库与历史最快匹配frozen base artifact，不称Private是
所有可用路线的全局最快。反序/安静窗口、LoRA generation盈利、更多
scene/seed/encoder、Public/Private operation及真正GGUF ahead-decode
仍待推进，完整目标保持active。

完整cold/warm、phase/calls/source、timings/memory、PNG/库哈希、失败与
清理证据见 [机器记录](../design/validation/local512-qwen-generation-phases-20261010.json)。
