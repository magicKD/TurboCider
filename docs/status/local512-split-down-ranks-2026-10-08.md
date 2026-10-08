# Qwen512²：分片 down-LoRA ranks 的真实运行与负/小幅结果

2026-10-08，Asia/Singapore，M4 Max64GB / macOS26.6.2。接续
[shared ranks / GPU / decode](local512-shared-lora-and-decode-2026-10-08.md)及
[native W8 surfaces](local512-native-weight-surfaces-2026-10-08.md)。只读本地
模型与真实 adapter，无下载、权重重写或磁盘 dense 副本。完整 Z/Qwen
base、真实 LoRA、1–2参考图、encoder、GGUF/ConvRot及盈利目标仍 active。

## 实现：提前计算 GPU hidden 的 down-A，而不是漏掉 down-LoRA

`TURBOCIDER_QWEN21_RUNTIME_SPLIT_DOWN_RANKS=1` 默认关闭。GPU corrected
hidden 的 `H_gpu A_gpuᵀ` 与原 GPU head 一起提交；恢复 ANE hidden 后，
计算 `H_ane A_aneᵀ`，FP32 相加，再完成一次 B projection、一次原 delta
dtype 舍入、一次 base add。无需先 concatenate 完整 hidden。此改变会
改变 FP32 reduction order；不称数学/字节逐位等价，不重新量化或 merge
LoRA，不删除 alpha、正/负 stacked adapters 或 hidden dtype 边界。

新的 metadata-only delta API 复用原 adapter implementation；ranks 和
compiled factories 都绑定当前请求的同一 resident Weights。没有 out
adapter 的 block 使用原 callback。裸 rank vectors 不自动识别同 shape
跨 adapter generation，调用方不得跨 input/range/generation 复用。

HybridFfn 可选 hook 只返回 lazy arrays，由 wrapper 提交及 drain。prepare/
finish 抛错和错误 dtype 保留原错误、drain GPU/ANE并可恢复；晚期 chunk
失败完整 GPU FFN 重算，不能发布部分结果。保留原 full-hidden scratch
保守上界，另加每行 `3 * total_rank_width * 4` bytes，没有放宽 admission。
实际成功返回后才增加 split-down blocks/arrays 两个 session counters。

flag 必须0/1；active route 要求 approximation、resident512²、真实
runtime LoRA、Private fixed nonzero channels及 FP16 rank option 关闭。
普通 GPU/base 的合法 flag 不起作用，以便完整 GPU 控制使用相同环境。
本轮未将该 hook 自动用于 Public row route、Z LoRA 或 calibration 搜索。

## 实模型：两图局部较快，一图变慢，仍未超过完整 GPU

同一 Private v1 library、本地 Viggle r256、512²、六步/seed29，各三个
fresh prompts（conditioning miss），相同 encoder source retention；GPU
encoder、Private DiT5120，两 hybrid arm 都共享 gate/up ranks，W-code
cache关闭。每个 arm 独立进程一冷两热，100ms process-tree采样，无
continuous-load 正式资格，`complete_diagnostic / qualification_passed=false`。

| workload / order | complete GPU warm s | original hybrid warm s | split ranks warm s |
| --- | ---: | ---: | ---: |
| 一图，on→off→GPU | 10.968586 | 11.709273 | 12.045587 |
| 两图，GPU→off→on | 13.119773 | 14.495052 | 14.050064 |

一图 on 比 off 慢2.87%，比 GPU 慢9.82%；两图 on 比 off 快3.07%，但
仍比 GPU 慢7.09%。不是双向重复或稳定速度资格，不能合并不同 workload
取得有利分母。cold on/off/GPU：一图17.314612/16.236879/15.845076s；
两图18.279603/18.433158/17.855520s。保留全部样本及负结果，默认不提升。

on 实际 split blocks/arrays 累计各192/384/576；off各0。每请求一图224、
两图256 actual ANE calls；failure/fallback/retry0，encoder source只load1次。
完成计数不是 selection marker，也不证明物理 INT8 MAC 或 GPU/ANE overlap。

六个 memory reports complete，system swap-in/out0；一图 on/off/GPU
phys-footprint peak约38.65/38.30/40.06GB，两图 GPU/off/on约40.83/38.89/
39.24GB。范围为进程树 load+cold+warm+exit，不包含外部服务/driver归因。
保守 scratch 并非实测物理 RAM cap。

## 画面与数值边界

六个 off/on PNG pair 并非 byte exact。已目视全部六个原尺寸 whole pair
及每组 center/top-left/bottom-right 三同坐标裁剪：壶形、盖/钮、把手、
布局、釉色、桌面和阴影非常接近，釉面细纹/高光稍有变化，未见新增
断裂或色块。这只是本 scene/seed 的有限 agent 观察，不是用户批准、
多seed质量或对 GPU oracle 的完整资格。视觉工具固定标题写 GPU reference，
本轮左图实际是 **original Private hybrid**；不将它误报为纯 GPU 对照。
自动 manifest 保持 pending，历史 N1/数值失败不改写。

保留目录：
`outputs/local512-qwen-edit{1,2}-down-rank-v1-diagnostic-20261008/`，每组
`visual-{0,1,2}` 内有 whole、三裁剪与 provenance manifest。

## 回归、构建与提交范围

独立 math18 cases：FP32/FP16/BF16、rows1/17/67、两种 channel cuts、
stacked正/负/alpha；F32 delta relL2<1e-5，compiled final add<.002。
实际 Private channel rows17/67原/split对照，prepare/finish/wrong-dtype
错误保留与恢复、晚期 chunk失败 complete-GPU marker，全部通过。
request gates和 forged/missing/reset successful counter工具测试通过。
最终 Public19项、Private22项 selected regressions均通过、无skip；包含
原 project/shared ranks、实际 encoder、Private channel/native calibration。
不是全仓或 Public 实模型加速验收。新 gate夹具首次遗漏移除 base 的
inference_time strategy导致合法拒绝；已仅修正夹具，原失败log保留。

Public actual release-binary guard通过。两库各491source input hashes均
匹配当时工作树，含原 ConvRot草稿，是 working-tree snapshots，而非
clean staged-only build。Private native-only build exit0；Public native
库/CLI和manifest验证完成后误进入额外 Swift/App 阶段，仅停止确认的
owned build进程，外部服务未触碰；不称整个 Public build exit0或App完成。

```text
Private e4f77f3b8ad7c2e38408ba214652c9e9ec9546b744318c3fb7986d72b8f7df06
Public  96e42d71c3f0ec51270bbc3211a09f7f10c3a3449071109aa229b9d3397ccde0
```

所有本轮 owned jobs终止后，清理425个可重建`.o`、33,075,624 logical
bytes及两个 module-cache（Public约332,152KiB，Private空）。保留库/CLI/
probes、log、manifest/PNG、模型及 adapter，无大 activation dumps。
选择性提交只含本轮 owned hunks，原 ConvRot drafts保持原样。
相对路径机器记录见
[本轮证据](../design/validation/local512-split-down-ranks-20261008.json)。

接续重点仍是 GPU packed/融合 consumer、实际 GGUF decode复用/eviction、
Qwen盈利分区与完整GPU graph边界、Public/Private按operation不可变选择、
敏感激活 handoff、多prompt/seed与有效 strict窗口；不能用这些兼容API
和局部两图收益代替完整加速目标。
