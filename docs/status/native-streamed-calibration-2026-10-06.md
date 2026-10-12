# 有界 GPU 校准与预算内实测通道点

UTC/Asia-Singapore 2026-10-06。接续 `b5ca847`。完整双后端、Z/Qwen
两分辨率 base ≥1.2×、实 LoRA 加速及质量/内存/device trace 目标仍
active。本轮缩减的是校准 GPU retention，不是扩大内存额度或改变
ANE 算术；不宣布完整目标完成。

## 实现与约束

LoRA 校准改用 `submit_streamed`：每层实际生成 range corrections、
W/A staging、提交独立 GPU upload/restore、完成 GPU head，再 join
完整 hidden 并执行一次 down-LoRA。传输、head 和 join 全部完成后
才重用共享 GPU tail/hidden targets、释放当层 correction owners。
GPU 侧不再同时保留四层 correction pairs、full hidden 与 joins。
四组 immutable ANE W/DG/DU/Y bindings 和独立 frozen restore copies
仍保留；GPU-alone 不读本次 live ANE output，不改变独立 one/four
ANE 测量。旧 base/static/batched 路径不删。

部分 producer/head/fence/drain/join 异常仍保留首个错误，所有尝试过
的 producers 恰好一次排空；不发布成功 completion。晚 join 抛错
也调用其 fence，健康 refill 和析构 correction cleanup 仍可用。
共享 mutable targets 有实 GPU 测试：四层不同 base-down 和 hidden
快照逐层恢复，对独立 oracles 检查，不能提前覆盖下一层读者。

新增 shared host-only `plan_native_channel_calibration_memory`，把
当前/双槽 W/A/correction、四组 ANE banks/outputs、冻结快照、GPU
head scratch、恢复目标、input/padded input 和固定 allowance 放进
同一 admission estimate。流式路径只减去已真实释放的 retention：
三份已完成 head result、三份 correction/full-hidden/join 工作集及
三套 GPU restore targets；保留 projection scratch 项和原 allowances。
Surface 按64-byte pitch、实际 page size逐个舍入，fixture覆盖4/16KiB。
这是 opportunistic payload estimate，不是 process/driver RAM 硬上限。

`plan_channel_sampling` 优先原约0.4/0.8点。其完整工作区无法同时准入
时，选两个已独立 admission 的512-aligned点；不足两点则 GPU-only。
初筛用同一次 Mach observation，构造每个实际 arena 时重新观察并
准入，真实 inference executor 仍重新做 admission/self-test。fit
只在已测范围选 share，不将小点外推回未测的0.4/0.8。预算、system
reserve、5%候选窗口收益与原 numerical gate 全部不变。

graph calibration ABI 新增 `prepared-channel-streamed-gpu-v2-b…`，与
旧批量 sampling/cache 隔离；ANE math recipe 本身仍 v2-lora。raw
report 导出 `gpu_retention_layers`、`memory_limited_points` 及每个
preflight candidate 的 byte分项、总estimate、optional cap、实际
headroom/admission/reason。verifier检查分项合计、预算/范围、ABI/
retention一致性，以及每个流式实测点确有对应 complete admission。
cached receipt 保存原 preflight observation，不冒称为新的物理内存测量。

## 实模型 LoRA

新 v1 Private CLI/library、原 BF16 model，fox/seed42、strength1，
inference-time LoRA。Z distill patch8步；Qwen Viggle rank256六步，原
FP32 rank matmuls。1056/4224-row activation-input模板、k1024/n512、
chunks1/fixed-async1/scale-cache1/launch-fence1/stage-specialize1，
prefetch/A8-lookahead/deferred0。Qwen1024使用既有显式诊断flag。
四个模型 timed arms 在所有 owned build/tests结束后串行；连续 host
observer 开启，但没有正式 competing-load qualification/物理 trace。

### 1024：现在能测完两个点，不再在 arena 前全部拒绝

| 模型 | actual/bucket | 实测Fa | 对应完整estimate MiB | 最终采用 |
| --- | ---: | --- | --- | --- |
| Z | 4128/4224 | 1536、3072 | 1692.765625、2019.578125 | trial采用3072；实请求拒绝验收 |
| Qwen | 4096/4224 | 1024、2048 | 1731.921875、1954.671875 | 预测收益不足，GPU-only |

optional cap仍2048MiB；两模型均 `complete=true`、memory-limited点、
GPU retention1、每点90次实际 correction pair计算/180次上传。
raw one/four vectors及byte分项经verifier检查，不借用旧base slopes。
不把以上payload estimate当成实分配峰值或整个请求内存上限。

Z独立FFN trial：rel L2=0.0109484152、cosine=0.9999402715，four-FFN
GPU/candidate hot medians0.303778750/0.243508958s，36次calls，零
trial fallback/retry。**不是整请求1.2×证据**。实际生成冷请求仍有
2次overflow retry、headroom升到16、258次calls；热请求另256次。
冷/热PNG不同，recipe-drift gate正确拒绝，两者无最终failure/fallback。
冷/热request wall81.191946/36.938365s，未作同库GPU完整请求对照。
不能拿前轮GPU-only的40.065979s当分母。

Qwen独立full-GPU slope0.0983403887s；Fa1024/2048的GPU-side slopes
0.110936500/0.104888764s，已经不比完整GPU基线更快，fit正确拒绝
预测收益不足5%。无actual candidate trial，不伪造采用或收益；两次
生成Private calls=0、failure/retry/fallback=0。冷/热request wall
71.208564/34.590831s。PNG同为
`fab752c57e5ff5b90db9281c67fce6975ab0c0e731621a03358bfe56d66bd18b`，
与前轮同GPU-only固定样本一致，不是ANE质量证明。

### 512 回归

Qwen原点Fa5120/9728都可准入（1125.921875/1622.046875MiB），最终
Fa5120，trial rel L2=0.0130570005、cosine=0.9999149733，GPU/candidate
four-FFN medians0.107609209/0.093804292s。每请求192次真实Private
calls、零failure/retry/fallback、headroom1，request receipt contract
通过；冷/热wall28.243013/8.441746s。PNG仍为前轮
`9d5ac6e3a61123c977d514dd5b9e28f3957ff7fe3a331681ecdf103843ee07c7`。
不存在本轮matched/reverse整请求倍率或正式latent/媒体资格。

Z原点Fa4096/8192可准入（964.093750/1385.593750MiB），trial采用4096。
实际冷请求仍2次overflow retry/headroom16，258次calls，热请求另256次；
冷/热wall25.345054/7.237371s。PNG逐个与前轮对应cold/hot一致，
但两者本身不一致，因此两请求仍被recipe-drift gate拒绝。
本轮没有通过保持数值旧结果而放宽稳定性门槛。

## 构建、测试和原始证据

- 90项shared host/calibration/screen/load/image/release guards通过。
- Private三项MLX集成通过：显式LoRA兼容、native-auto正/负strength
  rank4与新streamed/memory ABI、prepared calibration的双share
  batched/streamed异常、shared restore、owner release/refill/析构测试。
- Public Core ML/MLX/receipt13项通过；新memory字段实际serialization
  assertion通过。actual public flags/private class bytes/direct-link发行
  隔离检查通过。
- Private全native/Swift App及集成测试**构建**结束exit0，不冒称所有
  编出来的Swift tests都已运行或完整make test绿色。Public单独library
  build。所有owned build/test在实模型timed arms前结束。

Private library SHA256：
`321f57e395616f63bb2d039ca5b13091f6f7982c76acc8b951820e97cedff3f3`。
Public：`09f58598cf6acfb7fe3dfcd4a47cb7932e87de1ea257d276e681a1832b66e0a5`。
Private CLI：`f09144aed59e0899e0f3cd06ec97e54958ac91094f7b54453aeb84faf62ea02f`。
native source manifests对当前tree无mismatch；仍是包含原ConvRot drafts
的working-tree build，不是clean staged-only全重建。只选择本轮results
receipt hunks；原Z/results draft changed-lines完全相同，staged receipt
syntax check通过。models/adapters/references/原设计文稿未改。

各目录 `outputs/native-streamed-channel-{q,z}{512,1024}-20261006/`
保留独立cold/hot请求与PNG，未覆盖前轮。`generation-v1-receipt.json`
含实际CLI hash、连续host观察、原始results与events，SHA256分别：

- q1024：`e5cdb883ef892f3addd415c64bae4d169af58047ea7f9e9f024921465da7f561`。
- z1024：`ca42929008fd8d963d421eeefb85af4444d17d056a61e207ceb03ecd4a7fba18`。
- q512：`023c69cbee123a95f8941846f3bcbf20a90deced011e82c9ff2f8ab7b9c8fa1a`。
- z512：`3d5f66f07e4eeea7aa5c223e6e1bd17af11ec7ec742fb2706c69f5c7e96f961b`。

## 下一步：真实GPU补集与敏感层

Qwen source inspection发现：校准的channel/adapter/down callbacks目前
是eager family算术，实际request-local LoRA callbacks已经mx::compile。
full-GPU calibration分母则已compiled。需要将校准GPU补集也与真实
compiled图完全对齐，再测1024；尚未用同binary A/B量化此差异，不能
把当前no-gain提升为所有优化后的ANE组合均无收益，也不能现在预报
会快1.2×。默认0.4/0.8的较大share若要重新开放，仍需进一步减少
真实arena或重新测memory-admitted算法，不能绕过cap或外推。

Z还需定位真实overflow所在层/phase，将敏感层GPU接手或对实际
headroom recipe独立校准、质量验证，不能仅删除拒绝门槛。随后接续
按operation/layer的Public/Private/GPU选择、GGUF真实有界reuse与
Comfy GPU handoff，以及原base四格matched/reverse/multi-prompt
≥1.2×、LoRA、最终latent/感知/语义、process memory与physical trace。
以上仍是完整目标的未完成项。
