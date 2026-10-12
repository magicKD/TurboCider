# 共用 compiled LoRA 图、逐层溢出诊断与敏感层 GPU 路由

UTC/Asia-Singapore 2026-10-06，接续 `c323036`。原目标仍 active：双
后端完整集成、Z/Qwen 两分辨率 base ≥1.2×、真实 LoRA 加速，以及
latent/媒体、process memory、physical trace 资格均不能由本轮替代。

## Qwen：校准与真实 compiled 图统一

新增 `qwen21/runtime_ffn_graphs.hpp`，full GPU、logical gate/up range
corrections、physical GPU channel head、ONE-full-hidden down-LoRA 共用
request/calibration-local factories。保留原投影、FP32 rank/累加、BF16
delta/最终舍入及 checkpoint-only partial down；不合并 master weights，
不新建常驻 compact projection，不全局缓存 adapter captures。

实际请求原来已 compiled；本轮将校准的 eager head/corrections/down
改成相同 compiled graph bodies。校准 range cache 按 ordinal/first/count
隔离，只活到 constructor 校准完成，不能跨 adapter rebinding。冷 trace
在每 cell 的 warmup 中完成，hot 保留七样本。配置 identity 显式标记
`shared-compiled-qwen-ffn-v1`，原数学 recipe 不改。

实 GPU fixture 比较新 factories 与原 request-local compiled bodies，
base/A/B/base（.75/-.5强度）、17/67行、两个物理范围、full/head/
corrections/down 输出逐位一致。旧 LoRA、失败全操作重算、取消、
deferred output lifetime 的既有兼容测试不删除。

最终 v2 库，Viggle rank256 strength1、6步、原FP32 rank、fox/seed42，
resident、1056/4224-row模板、k1024/n512，chunks1/fixed-async1、
scale-cache/launch-fence/stage-specialize1，prefetch/A8-lookahead/
deferred0。Qwen1024保留既有显式诊断flag。均完成两点90/180次实际
correction计算/上传，独立 FFN trial 零fallback/retry，各请求192次
Private calls、headroom1、零failure/retry/fallback，receipt contract通过。

| Qwen尺寸 | 自动Fa | trial rel L2 / cosine | four-FFN GPU/candidate median s | 冷/热 request wall s |
| --- | ---: | --- | --- | --- |
| 512 | 5120 | 0.0130570005 / 0.9999149733 | 0.105705209 / 0.090059000 | 25.312666 / 8.375441 |
| 1024 | 2048 | 0.0083631114 / 0.9999652948 | 0.410764458 / 0.384246250 | 79.683650 / 37.011976 |

1024此前 eager-calibration 未通过预测收益，本轮第一次真正采用 Fa2048。
这不代表整请求快1.2×，也不证明比旧库的GPU-only更快；不能用之前
34.590831s或不同library的GPU分母计算倍率。GPU part slopes目前
Fa1024/2048为0.098155208/0.088307917s，对独立full GPU slope
0.096913167s；actual trial仍另做完整 HybridFfn 窗口收益检查。
512/1024冷/热各自PNG一致，hash分别
`9d5ac6e3a61123c977d514dd5b9e28f3957ff7fe3a331681ecdf103843ee07c7` /
`1deca36b7a4eb171921540463c39c96670530b54f81f92e6327ee9146aadb229`。
512延续旧固定样本；1024已不是前轮GPU-only输出，不将一致性当成质量。

## 两后端共用的溢出回执

`RunResult.headroom_start_scale` 在真实 launch worker 读取实际值；
Public Core ML、Private FP16 CPU/device、Private W8A8 都填充。
W8A8失败路径也返回真实最终headroom，修复原先错误地回报默认1的
诊断问题，不改变重试/算术/失败策略。

`HybridMetrics` 新增固定32项、noexcept、无动态分配的 overflow prefix。
记录实际 layer ordinal、requested rows、runtime call begin/count、
retry count、headroom before/after 和完成状态；超出容量只增加dropped。
只聚合每次FFN launch，不伪称精确row chunk/设备时间/物理trace。
失败launch保留失败状态；非有限统计JSON为null，不美化为1/通过。

host verifier检查字段/有限性、calls无重叠、retry总量、bounded prefix/
dropped和resident prefix不变性。旧receipt全无此字段时保持显式兼容；
缺一部分字段或伪造physical trace拒绝。原native-auto的zero-retry/
headroom1 recipe-drift gate不删除。

v1未选择GPU敏感层的实模型诊断，两分辨率Z、distill patch strength1、
8步、相同fox/seed42，都只有两项overflow记录：

| ordinal / FFN源 | call begin / count | headroom | retry | 推导去噪步 |
| --- | --- | --- | ---: | --- |
| 2 / layers.0.feed_forward | 66 / 2 | 1→4 | 1 | 第3步 |
| 2 / layers.0.feed_forward | 195 / 2 | 4→16 | 1 | 第7步 |

步号由每步32个FFN及此前的1次retry复算；不是新增的直接timestep字段。
明确不是noise_refiner。512/1024分别Fa4096/3072，cold258次calls、hot
另256次，冷/热PNG不同；两请求仍被recipe-drift gate拒绝。
原始负结果及原图保留，不以更好的热图/某一步替代它们。

## 显式敏感层 GPU：两个分辨率消除重试

共用 `HybridFfn::set_gpu_layers` 接受规范化、无重复、有限序号列表，
只在idle操作边界变更；invalid/mid-operation更新不修改当前policy。
被指定的block返回完整GPU plan，不走FFN bridge、不staging，也不
预取该层；使用 family 已有完整 GPU block，包括完整 LoRA。不是
错误fallback，也不在public/private间额外复制一套weights。移除policy
后可恢复hybrid，Public/Private都有实际接口回归。

Z新增显式 `TURBOCIDER_Z_RUNTIME_GPU_FFN_BLOCKS=2`，仅runtime
gpu_ane generation适用，严格解析0…31列表、在模型准备前拒绝非法
配置；未设置时行为不变。原manifest/executor identity和calibration
GPU configuration包含规范化policy。此列表名为FFN ordinal，但实际
选择family完整GPU block，不能描述成只计算一个低秩修正。
它仍是诊断/质量路由选项，尚未提升默认或按其他prompt自动推断。

最终v2、上述同一Z条件，GPU ordinal2，每次8个完整GPU blocks，其余
248次实际Private调用；cold/hot累计分别248/496，forced counts8/16。
两个分辨率都zero failure/retry/fallback、headroom1、overflow prefix为空。
actual routing/counters与native-auto receipt gates通过，warmup不被
选择性忽略。预算仍2GiB，最终ANE Fa为512的4096、1024的3072。

| Z尺寸 | 冷/热request wall s | 冷/热PNG（各自相同） |
| --- | --- | --- |
| 512 | 22.489779 / 7.274077 | 663b587147506c1c8420b36b6652012f67c72d9b3975c531c1ddb59b5bedbd36 |
| 1024 | 66.198559 / 33.466343 | 5491f929da0541b80a62aaac41649f2231ed3a9220a8a2a7f5803cd836923aa2 |

这是固定样本稳定性与实际融合路径，不是GPU-vs-W8A8最终latent/
感知/语义资格。Native质量validation calls=0不当验收；所有timed模型
arms串行且与owned build/test不重叠，有连续host observer，但没有正式
competing-load/reverse/multi-prompt或physical trace，不计算整模型倍率。

## 构建与证据

97项shared host/calibration/screen/load/image/release guard通过；新host
policy测试最初因range-loop复制的-Werror失败，改const reference后
重跑完整suite通过，不降低编译警告。Public Core ML/MLX/receipt13项
通过，又实际运行Public GPU policy子测试。Private3项driver测试和
2项MLX channel/auto集成通过，涵盖实际headroom起止、失败仍保留值、
shared compiled parity、GPU override无staging、移除/操作中禁止更新等。
不是完整make test或Swift全套验收。

最终v2 Private library SHA256：
`f6197b29aba7b8922bed25ad51d5470a3c87c50667c7ce3394494b52c8b2035a`。
Public：`03bfaf7518b43277473dc61bfee940dc87532e0206e15ba6eaddd716e6e4dc08`。
thin CLI SHA256仍`02c55923be1dff7fc8dfd5681fd26b1cfaf15d8553eec28d1d9ce09b16d11ebb`；
动态相邻library不同，不能只凭CLI hash称为同一runtime。实际runtime
build ID和library hashes另绑定。Public actual发行隔离检查通过。
v2 native source manifests对当前tree无mismatch；包含原dirty ConvRot
drafts的working-tree builds，不冒称clean staged-only重建。

原始final回执hash：

- `outputs/native-final-compiled-q512-20261006/generation-v2-receipt.json`：
  `a40c8da2091b9d90e71c017d7c2d1808dc025feef2ce3d727197d4b4be99c347`。
- 同前缀q1024：`d5035ad63c2a4eae18b8e9e8f61388adb4f617ba29a261e2b54aac790c7c68af`。
- `outputs/native-sensitive-layer-z512-20261006/generation-v2-receipt.json`：
  `4e73eb4b02cc1ac26c456c393808adce9c3dc165e10e2dec805a0807603769b1`。
- 同前缀z1024：`7f0db9310533576f55369281dc5a2dc2d88005c9add24329c4569db1ef438f1f`。

早期v1库是引入GPU policy之前的独立snapshot：Private
`6ad3f8badfc8a3aff59ed8cd87ed652eaf69d6701937a7aabdaed0dd019b292a`、
Public `e674729e0758c1ead9ebe4ac1563de75609cb79be35434322bb57f6b2983704d`。
其4格raw diagnosis在`outputs/native-compiled-channel-{q,z}{512,1024}-20261006/`；
Z负overflow日志来自此库，不能混作v2策略图像/最终性能证据。所有
目录独立，未覆盖旧PNG/回执；models/adapters/references/原设计文稿未改，
selective staging逐项保留原Z/results/session dirty changed-lines。

后续仍需最终同库optimized-GPU对照、反序/多提示词、完整latent/媒体/
内存/device trace、base四格≥1.2×；敏感层策略要扩展质量验证，Public/
Private/GPU每operation选择、GGUF真实有界reuse与Comfy敏感GPU handoff
继续接续，不能将本轮稳定性与FFN窗口收益替代完整目标。
